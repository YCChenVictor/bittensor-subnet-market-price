import pytest
from unittest.mock import MagicMock, AsyncMock
from market_price.validator import execute_forward


@pytest.mark.asyncio
async def test_execute_forward(mocker):
    # Mock time.sleep
    mocker.patch("time.sleep", return_value=0)
    # Mock bittensor.logging.info
    mock_logging_info = mocker.patch("bittensor.logging.info")
    # Mock get_rewards
    # (it is to mock the place calling the get_rewards, not the get_rewards itself)
    mock_get_rewards = mocker.patch(
        "market_price.validator.forward.get_rewards", return_value=[1, 1, 1]
    )

    # Can extract all the mocks in one file
    # mock self, this self comes from BaseValidatorNeuron
    mock_self = MagicMock()
    # The number of miners to query in a single step
    mock_self.config.neuron.sample_size = 3
    # The number of miners in the network
    mock_self.metagraph.n.item.return_value = 5
    # Mock vpermit_tao_limit
    mock_self.config.neuron.vpermit_tao_limit = 10
    # Mock metagraph
    mock_self.metagraph.S = {0: 5, 1: 10, 2: 15, 3: 20, 4: 25}
    # Mock dendrite to async and return responses
    mock_self.dendrite = AsyncMock(
        return_value=[
            {
                "movement_prediction": 0.001,
                "target_symbol": "symbol_1",
            },
            {
                "movement_prediction": 0.002,
                "target_symbol": "symbol_2",
            },
            {
                "movement_prediction": 0.003,
                "target_symbol": "symbol_3",
            },
        ]
    )

    # Call the function you want to test
    await execute_forward(mock_self)

    # Add your assertions here
    mock_get_rewards.assert_called_once()
    mock_logging_info.assert_any_call(
        f"Received responses: {[
            {
                "movement_prediction": 0.001,
                "target_symbol": "symbol_1",
            },
            {
                "movement_prediction": 0.002,
                "target_symbol": "symbol_2",
            },
            {
                "movement_prediction": 0.003,
                "target_symbol": "symbol_3",
            }
        ]}"
    )
    mock_logging_info.assert_any_call("Scored responses: [1, 1, 1]")
