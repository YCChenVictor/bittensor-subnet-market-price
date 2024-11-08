import pytest
from unittest.mock import Mock
from neurons.miner import Miner
from market_price.protocol import MarketPriceSynapse  # Adjust import as needed


@pytest.mark.asyncio
async def test_forward(mocker):
    # Create a mock instance of Miner without running __init__
    # should be removed create the wallet for testing
    miner_instance = Mock(spec=Miner)

    # Mock the predict method to avoid third-party calls
    miner_instance.predict = Mock(return_value=0.01)

    # Set model_config directly on the mock instance to avoid file I/O
    miner_instance.model_config = {"predict_symbol": "mock_symbol"}

    # Define a mock synapse object with the needed attributes
    synapse = MarketPriceSynapse(timestamp=1234567890)

    # Attach the actual forward method to the mock instance to test its logic
    miner_instance.forward = Miner.forward.__get__(miner_instance, Miner)

    # Call forward with synapse, allowing forward's logic to run
    result = await miner_instance.forward(synapse)

    # Assert the expected behavior in the synapse after forward executes
    assert result.movement_prediction == 0.01
    assert result.target_symbol == "mock_symbol"
