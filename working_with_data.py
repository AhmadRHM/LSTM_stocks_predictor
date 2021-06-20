import pytse_client as tse
import numpy as np


def main():
    # raw_data = tse.download(symbols="all", write_to_csv=True)

    closed_prices, stocks_info = [], []
    for symbol in tse.all_symbols():
        ticker = tse.Ticker(symbol)
        closed_prices.append(ticker.history.close.to_numpy())
        print(symbol, "added")
        stocks_info.append(np.array([ticker.base_volume]))
    return closed_prices, stocks_info


if __name__ == '__main__':
    closed_prices, stochs_info = main()

