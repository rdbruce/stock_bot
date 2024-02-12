from dataclasses import dataclass
from ruamel.yaml import YAML

yaml = YAML(typ='safe')

@yaml.register_class
@dataclass
class StockConfig:
  ticker_list: list
  ticker_to_predict: str
  window_size: int
  test_days: int
  num_days_concerned: int
  batch_size: int
  epochs: int
  trading_quanta: float
  api_key: str
  secret_api_key: str


def get_stock_config(fname):
    with open(fname, 'r') as fh:
        return yaml.load(fh)
