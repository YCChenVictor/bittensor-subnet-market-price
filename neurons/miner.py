# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the “Software”), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import time
import typing
import bittensor as bt
import asyncio
import json
import pandas as pd
import numpy as np
import tensorflow as tf

import market_price

# import base miner class which takes care of most of the boilerplate
from market_price.base.miner import BaseMinerNeuron
from model.market_price_movement_prediction.scrape_finance_data_yahoo import (
    scrape_and_save_data,
)
from model.market_price_movement_prediction.etl import ETL
from multi_time_series_connectedness import (
    Volatility,
    RollingConnectedness,
)


class Miner(BaseMinerNeuron):
    """
    TODO: overwrite blacklist and priority functions

    This class inherits from the BaseMinerNeuron class, which in turn inherits from
    BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up
    wallet, subtensor, metagraph, logging directory, parsing config, etc. You can
    override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting
    unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests
    to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        with open("model_config.json", "r") as file:
            self.model_config = json.load(file)

    def predict(self, timestamp: int):
        print("scraping finance data for predicting")
        asyncio.run(
            scrape_and_save_data(
                self.model_config["train_symbols"],
                self.model_config["prices_predict_dir"],
            )
        )

        print("modifying data")
        past_roll_conn_period = self.model_config["past_roll_conn_period"]
        periods_per_volatility = self.model_config["periods_per_volatility"]
        volatilities_from = (
            timestamp - (past_roll_conn_period + periods_per_volatility + 1) * 60
        )
        volatilities_to = timestamp
        etl = ETL(
            self.model_config["prices_predict_dir"],
            self.model_config["washed_predict_dir"],
        )
        etl.load_data()
        etl.transform_into_same_timestamp(volatilities_from, volatilities_to)

        print("calculating volatilities")
        volatility = Volatility(n=2)
        predict_dir = self.model_config["predict_dir"]
        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir)
        volatility.calculate(
            self.model_config["washed_predict_dir"],
            f"{predict_dir}/volatilities.pickle",
        )

        print("calculate rolling connectedness")
        volatilities = pd.read_pickle(f"{predict_dir}/volatilities.pickle")
        roll_conn = RollingConnectedness(
            volatilities.dropna(),
            self.model_config["max_lag"],
            periods_per_volatility,
        )
        roll_conn.calculate(f"{predict_dir}/roll_conn.pickle")

        print("predict movements")
        with open(f"{predict_dir}/roll_conn.pickle", "rb") as f:
            predict_roll_conn = pd.read_pickle(f)
        columns_to_remove = [
            "start_at",
            "end_at",
            "forecast_at_next_period",
            "forecast_at",
        ]
        input_data = predict_roll_conn.drop(columns=columns_to_remove).values
        input_data = np.expand_dims(input_data, axis=0)
        model = tf.keras.models.load_model("trained_model.keras")
        prediction = model.predict(input_data)
        return prediction

    async def forward(
        self, synapse: market_price.protocol.MarketPriceSynapse
    ) -> market_price.protocol.MarketPriceSynapse:
        """
        Processes the incoming synapse by performing a predefined operation on the
        input data.

        Args:
            synapse (template.protocol.MarketPriceSynapse): The synapse object
            containing the input data.

        Returns:
            template.protocol.Dummy: The synapse object with the output field populated.
        """
        timestamp = synapse.timestamp
        synapse.movement_prediction = self.predict(timestamp)
        synapse.target_symbol = self.model_config["predict_symbol"]
        return synapse

    async def blacklist(
        self, synapse: market_price.protocol.MarketPriceSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        Your implementation should define the logic for blacklisting requests based on
        your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before
        synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is
        important to blacklist requests before they are deserialized to avoid wasting
        resources on requests that will be ignored.

        Args:
            synapse (template.protocol.MarketPriceSynapse): A synapse object
            constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the
            synapse's hotkey is blacklisted, and a string providing the reason for the
            decision.

        This function is a security measure to prevent resource wastage on undesired
        requests. It should be enhanced to include checks against the metagraph for
        entity registration, validator status, and sufficient stake before
        deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient
        - stake.

        In practice it would be wise to blacklist requests from entities that are not
        validators, or do not have enough stake. This can be checked via metagraph.S
        and metagraph.validator_permit. You can always attain the uid of the sender via
        a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow
            # requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey "
                    f"{synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(
        self, synapse: market_price.protocol.MarketPriceSynapse
    ) -> float:
        """
        The priority function determines the order in which requests are handled. More
        valuable or higher-priority requests are processed before others. You should
        design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling
        entity's stake in the metagraph.

        Args:
            synapse (template.protocol.MarketPriceSynapse): The synapse object that
            contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function
        determines which request should be processed first. Higher values indicate that
        the request should be processed first. Lower values indicate that the request
        should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority

    def save_state(self):
        pass


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
