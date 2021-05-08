from __future__ import division
import pandas as pd
from pandas_datareader.data import DataReader
import numpy as np
import matplotlib.pyplot as plt
import pprint

def get_symbols(tickers):
    """Retrieve ticker price data from FRED."""
    portfolio_df = []
    for ticker in tickers:
        print(f"Getting {ticker} data...")
        portfolio_df.append(DataReader(ticker, 'fred', "2018-01-02", "2021-04-05"))
    portfolio_df = pd.concat([bond_yields for bond_yields in portfolio_df], axis=1).dropna()
    portfolio_df.to_csv("bond_data.csv")


def load_yields():
    """Load stock data from csv."""
    return pd.read_csv("bond_data.csv")


def find_dates(bond_yields_historic):
    """Question 2.1"""
    dates = []
    #  Period 1.
    print("Period 1 \n")
    period_1 = bond_yields_historic.loc[(bond_yields_historic['DGS2'] == 2.88) &
                                        (bond_yields_historic['DGS5'] == 3.07) &
                                        (bond_yields_historic['DGS10'] == 3.23)]
    dates.append(period_1["DATE"])
    print(period_1)
    period_2 = bond_yields_historic.loc[(bond_yields_historic['DGS2'] == 1.47) |
                                        (bond_yields_historic['DGS5'] == 1.4) |
                                        (bond_yields_historic['DGS10'] == 1.56)]
    dates.append(period_2)
    print("Period 2 \n\n")
    print(period_2)
    period_3 = bond_yields_historic.loc[(bond_yields_historic['DGS2'] == 0.31) |
                                        (bond_yields_historic['DGS5'] == 0.46) |
                                        (bond_yields_historic['DGS10'] == 0.85)]
    dates.append(period_3)
    print("Period 3 \n\n")
    print(period_3)
    period_4 = bond_yields_historic.loc[(bond_yields_historic['DGS2'] == 0.15) |
                                        (bond_yields_historic['DGS5'] == 0.36) |
                                        (bond_yields_historic['DGS10'] == 0.82)]
    dates.append(period_4)
    print("Period 4 \n\n")
    print(period_4)
    period_5 = bond_yields_historic.loc[(bond_yields_historic['DGS2'] == 0.16) |
                                        (bond_yields_historic['DGS5'] == 0.87) |
                                        (bond_yields_historic['DGS10'] == 1.66)]
    dates.append(period_5)
    print("Period 5 \n\n")
    print(period_5)
    return dates


def get_bond_price(bond_yield, coupon, maturity, par=100):
    #  Assume coupon paid annually
    coupon /= 100
    bond_yield /= 100
    cf_t = par * coupon
    bond_price = 0
    for i in range(1, maturity+1):
        bond_price += cf_t / ((1 + bond_yield) ** i)
    return bond_price + (par / (1 + bond_yield) ** maturity)


def compare_bond_prices(bond_prices):
    given_prices = [99.74, 99.11, 97, 92.47, 100.06, 100.5, 100.55, 104.28, 101.57, 103.25,
                    106.2, 114.17, 99.95, 99.45, 98.19, 94.69, 99.93, 99.42, 95.19, 90.48]
    #  Generate plot to check for perfect linear relationship.
    m, b = np.polyfit(bond_prices, given_prices, 1)
    plt.plot(bond_prices, given_prices, "o")
    plt.plot(bond_prices, m*np.array(bond_prices) + b)
    plt.xlabel("Calculated Bond Prices")
    plt.ylabel("Given Bond Prices")
    plt.show()


def get_duration(price, coupon, maturity, bond_yield):
    coupon /= 100
    bond_yield /= 100
    cf_t = coupon * 100
    duration = 0
    w_t = 0
    for i in range(1, maturity+1):
        w_t = (cf_t / ((1 + bond_yield) ** i)) / price
        duration += i * w_t
    return round(duration, 4)


def goalseek(data):
    implied_yields = []
    for bond in data.keys():
        # print(f"Working on bond {bond}.")
        actual_price = data[bond][0]
        coupon = data[bond][1]
        maturity = data[bond][2]
        b_yield = 2
        est_price = get_bond_price(b_yield, coupon, maturity)
        while not g_y(est_price, actual_price):
            if est_price > actual_price:
                b_yield += 0.02
            else:
                b_yield -= 0.02
            est_price = get_bond_price(b_yield, coupon, maturity)
        implied_yields.append(round(b_yield, 3))
    return implied_yields


def g_y(computed_price, given_price):
    """check if computed price is almost equal to given price."""
    return abs(computed_price - given_price) <= 0.2


def compare_yields(implied_yields):
    given_yields = [2.88, 3.07, 3.23, 3.4, 1.47, 1.4, 1.56, 2.06, 0.31, 0.46, 0.85,
                    1.42, 0.15, 0.36, 0.82, 1.6, 0.16, 0.87, 1.66, 2.32]
    #  Generate plot to check for perfect linear relationship.
    m, b = np.polyfit(implied_yields, given_yields, 1)
    plt.plot(implied_yields, given_yields, "o")
    plt.plot(implied_yields, m*np.array(implied_yields) + b)
    plt.xlabel("Implied Yield")
    plt.ylabel("Given Yield")
    plt.show()


if __name__ == "__main__":
    tickers = ["DGS2", "DGS5", "DGS10", "DGS30"]

    bond_yields = load_yields()
    # 2.1
    # period_dates = find_dates(bond_yields)
    # print(period_dates)

    # 2.2
    #  Each bond denoted by an index, list = [yield, coupon, maturity].
    # bonds_to_price = {1: [2.88, 2.75, 2],
    #                   2: [3.07, 2.88, 5],
    #                   3: [3.23, 2.88, 10],
    #                   4: [3.4, 3, 30],
    #                   5: [1.47, 1.5, 2],
    #                   6: [1.4, 1.5, 5],
    #                   7: [1.56, 1.63, 10],
    #                   8: [2.06, 2.25, 30],
    #                   9: [0.31, 1.13, 2],
    #                   10: [0.46, 1.13, 5],
    #                   11: [0.85, 1.5, 10],
    #                   12: [1.42, 2, 30],
    #                   13: [0.15, 0.13, 2],
    #                   14: [0.36, 0.25, 5],
    #                   15: [0.82, 0.63, 10],
    #                   16: [1.6, 1.38, 30],
    #                   17: [0.16, 0.13, 2],
    #                   18: [0.87, 0.75, 5],
    #                   19: [1.66, 1.13, 10],
    #                   20: [2.32, 1.88, 30]}

    # print(get_bond_price(bonds_to_price[1][0], bonds_to_price[1][1], bonds_to_price[1][2]))

    # calculated_bond_prices = []
    #
    # for bond in bonds_to_price.values():
    #     calculated_bond_prices.append(get_bond_price(bond[0], bond[1], bond[2]))
    # compare_bond_prices(calculated_bond_prices)

    # 2.3:
    # [givenPrice, coupon, maturity]
    bond_data = {1: [99.74, 2.75, 2],
                 2: [99.11, 2.88, 5],
                 3: [97, 2.88, 10],
                 4: [92.47, 3, 30],
                 5: [100.06, 1.5, 2],
                 6: [100.5, 1.5, 5],
                 7: [100.55, 1.63, 10],
                 8: [104.28, 2.25, 30],
                 9: [101.57, 1.13, 2],
                 10: [103.25, 1.13, 5],
                 11: [106.2, 1.5, 10],
                 12: [114.17, 2, 30],
                 13: [99.95, 0.13, 2],
                 14: [99.45, 0.25, 5],
                 15: [98.19, 0.63, 10],
                 16: [94.69, 1.38, 30],
                 17: [99.93, 0.13, 2],
                 18: [99.42, 0.75, 5],
                 19: [95.19, 1.13, 10],
                 20: [90.48, 1.88, 30]}

    imp_yields = goalseek(bond_data)
    compare_yields(imp_yields)
    # 2.4
    # [price, coupon, maturity, yield]
    # bonds_to_get_duration = {1: [99.74, 2.75, 2, 2.88],
    #                       2: [99.11, 2.88, 5, 3.07],
    #                       3: [97, 2.88, 10, 3.23],
    #                       4: [92.47, 3, 30, 3.4],
    #                       5: [100.06, 1.5, 2, 1.47],
    #                       6: [100.5, 1.5, 5, 1.4],
    #                       7: [100.55, 1.63, 10, 1.56],
    #                       8: [104.28, 2.25, 30, 2.06],
    #                       9: [101.57, 1.13, 2, 0.31],
    #                       10: [103.25, 1.13, 5, 0.46],
    #                       11: [106.2, 1.5, 10, 0.85],
    #                       12: [114.17, 2, 30, 1.42],
    #                       13: [99.95, 0.13, 2, 0.15],
    #                       14: [99.45, 0.25, 5, 0.36],
    #                       15: [98.19, 0.63, 10, 0.82],
    #                       16: [94.69, 1.38, 30, 1.6],
    #                       17: [99.93, 0.13, 2, 0.16],
    #                       18: [99.42, 0.75, 5, 0.87],
    #                       19: [95.19, 1.13, 10, 1.66],
    #                       20: [90.48, 1.88, 30, 2.32]}
    
    # calculated_bond_durations = {}
    # for bond in bonds_to_get_duration.keys():
    #     cur = bonds_to_get_duration[bond]
    #     calculated_bond_durations[bond] = get_duration(cur[0], cur[1], cur[2], cur[3])
    # print(calculated_bond_durations)
    #
    # maturities = [2, 5, 10, 30]
    # print("Maturity = 2:\n")
    # print(calculated_bond_durations[1])
    # print(calculated_bond_durations[5])
    # print(calculated_bond_durations[9])
    # print(calculated_bond_durations[13])
    # print(calculated_bond_durations[17])
    #
    # print("\nMaturity = 5:\n")
    # print(calculated_bond_durations[2])
    # print(calculated_bond_durations[6])
    # print(calculated_bond_durations[10])
    # print(calculated_bond_durations[14])
    # print(calculated_bond_durations[18])
    #
    # print("\nMaturity = 10:\n")
    # print(calculated_bond_durations[3])
    # print(calculated_bond_durations[7])
    # print(calculated_bond_durations[11])
    # print(calculated_bond_durations[15])
    # print(calculated_bond_durations[19])
    #
    # print("\nMaturity = 30:\n")
    # print(calculated_bond_durations[4])
    # print(calculated_bond_durations[8])
    # print(calculated_bond_durations[12])
    # print(calculated_bond_durations[16])
    # print(calculated_bond_durations[20])

    # 2.5
    # new_yield_bonds = [bonds_to_price[bond] for bond in bonds_to_price if bond in [17, 18, 19, 20]]
    # print(new_yield_bonds)
    # print()
    # new_bond_prices = []
    # print("Bond 17")
    # print(get_bond_price(new_yield_bonds[0][0] + 0.25, new_yield_bonds[0][1],
    #                      new_yield_bonds[0][2]))
    # new_bond_prices.append(get_bond_price(new_yield_bonds[0][0] + 0.25, new_yield_bonds[0][1],
    #                                       new_yield_bonds[0][2]))
    # print("Bond 18")
    # print(get_bond_price(new_yield_bonds[1][0] + 0.25, new_yield_bonds[1][1],
    #                      new_yield_bonds[1][2]))
    # new_bond_prices.append(get_bond_price(new_yield_bonds[1][0] + 0.25, new_yield_bonds[1][1],
    #                                       new_yield_bonds[1][2]))
    # print("Bond 19")
    # print(get_bond_price(new_yield_bonds[2][0] + 0.25, new_yield_bonds[2][1],
    #                      new_yield_bonds[2][2]))
    # new_bond_prices.append(get_bond_price(new_yield_bonds[2][0] + 0.25, new_yield_bonds[2][1],
    #                                       new_yield_bonds[2][2]))
    # print("Bond 20")
    # print(get_bond_price(new_yield_bonds[3][0] + 0.25, new_yield_bonds[3][1],
    #                      new_yield_bonds[3][2]))
    # new_bond_prices.append(get_bond_price(new_yield_bonds[3][0] + 0.25, new_yield_bonds[3][1],
    #                                       new_yield_bonds[3][2]))
    # maturities = [2, 5, 10, 30]
    # old_bond_prices = [99.93, 99.42, 95.19, 90.48]
    #
    # plt.plot(maturities ,old_bond_prices , label="Old")
    #
    # plt.plot(maturities ,new_bond_prices , label="New")
    #
    # plt.xlabel("Maturity (Years)")
    # plt.ylabel("Bond Price ($)")
    #
    # plt.title("Effect of 25bps shift in yield curve")
    # plt.legend()
    # plt.show()
