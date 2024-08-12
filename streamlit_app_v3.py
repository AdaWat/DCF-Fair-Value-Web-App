import yfinance as yf
import pandas as pd
import streamlit as st
from statsmodels.tsa.api import Holt


def get_WACC(company):
    try:
        treasury_yield10 = yf.Ticker("^TNX")
        risk_free_rate = round(treasury_yield10.info.get('regularMarketPreviousClose') / 100, 2)
        sp500_teturn = 0.10
        beta = company.info.get("beta")
        cost_of_equity = round(risk_free_rate + beta * (sp500_teturn - risk_free_rate), 2)
        balance_sheet = company.balance_sheet
        financials = company.financials
        weight_of_equity = balance_sheet.loc["Stockholders Equity"][0] / balance_sheet.loc["Total Assets"][0]
        cost_of_debt = round(financials.loc["Interest Expense"][0] * -1 / balance_sheet.loc["Total Liabilities Net Minority Interest"][0], 2)
        weight_of_debt = round(balance_sheet.loc["Total Liabilities Net Minority Interest"][0] / balance_sheet.loc["Total Assets"][0], 2)
        tax_rate = round(financials.loc["Tax Provision"][0] / financials.loc["Pretax Income"][0], 2)
        WACC = round((weight_of_equity * cost_of_equity) + ((weight_of_debt * cost_of_debt) * (1 - tax_rate)), 2)
        return WACC

    except Exception:
        return 0.075


def least_squares_regression(data):
    n = data.size
    sigma_xy = 0

    for count, i in enumerate(data):
        sigma_xy += (n - count - 1) * i

    sigma_x2 = (n - 1) * n * (2 * n - 1) / 6
    sigma_x = (n - 1) * n / 2
    sigma_y = data.sum()

    slope = (n * sigma_xy - sigma_x * sigma_y) / (n * sigma_x2 - sigma_x ** 2)
    intercept = (sigma_y - slope * sigma_x) / n

    points = []
    for i in range(5):
        points.insert(0, slope * (i + 5) + intercept)

    forecast = pd.Series(points)
    forecast.index = [data.index[0] + pd.offsets.DateOffset(years=i) for i in range(5, 0, -1)]

    return forecast


def forecast(data):
    # forecast data ahead 5 periods, using Holt-Winters Damped Exponential Smoothing
    data = data.iloc[::-1]
    fit = Holt(data.to_numpy(), damped_trend=True, initialization_method="estimated").fit(smoothing_level=0.8, smoothing_trend=0.4)
    fcast = pd.Series(fit.forecast(5)).iloc[::-1]
    fcast.index = [data.index[-1] + pd.offsets.DateOffset(years=i) for i in range(5, 0, -1)]
    return fcast


def dcf(tick, requiredReturn, perpetualGrowthRate):
    data = pd.DataFrame({"freeCashFlow": tick.cashflow.loc["Free Cash Flow"], "pastRevenue": tick.financials.loc["Total Revenue"], "pastNetIncome": tick.financials.loc["Net Income"]})
    data = data.dropna().sort_index(ascending=False)  # remove nan values

    freeCashFlow = data["freeCashFlow"]
    pastRevenue = data["pastRevenue"]
    pastNetIncome = data["pastNetIncome"]
    pastNetIncomeMargins = data["pastNetIncome"] / (data["pastRevenue"] + 1e-10)

    projectedRevenue = forecast(pastRevenue)
    projectedNetIncomeMargins = forecast(pastNetIncomeMargins)
    projectedNetIncome = projectedRevenue * projectedNetIncomeMargins

    freeCashFlowRatesStdDev = (freeCashFlow / pastNetIncome).agg("std")
    if freeCashFlowRatesStdDev > 1:
        freeCashFlowRate = (freeCashFlow / pastNetIncome).median()
    else:
        freeCashFlowRate = (freeCashFlow / pastNetIncome).mean()

    projectedFreeCashFlow = projectedNetIncome * freeCashFlowRate
    sharesOutstanding = tick.info.get("sharesOutstanding")

    terminalValue = projectedFreeCashFlow.iloc[0] * (1 + perpetualGrowthRate) / (requiredReturn - perpetualGrowthRate)
    projectedFreeCashFlow = pd.concat([pd.Series(terminalValue), projectedFreeCashFlow])    # add terminalValue (it has index label 0)

    discountFactors = [[0, 0]] + [[freeCashFlow.index[0] + pd.offsets.DateOffset(years=i), 0] for i in range(5, 0, -1)]
    for count, i in enumerate(discountFactors):
        discountFactors[count][1] = (1 + requiredReturn) ** (len(discountFactors) - count)

    discountFactorsDataFrame = pd.DataFrame(discountFactors, columns=["timestamp", "discount"])
    discountFactors = pd.Series(discountFactorsDataFrame["discount"])
    discountFactors.index = discountFactorsDataFrame["timestamp"]

    presentValueOfFutureCashFlows = projectedFreeCashFlow / discountFactors
    presentValueOfCompany = presentValueOfFutureCashFlows.sum()

    fairValue = presentValueOfCompany / sharesOutstanding

    return max(fairValue, 0), freeCashFlow, projectedFreeCashFlow, pastRevenue, projectedRevenue, pastNetIncome, projectedNetIncome


st.set_page_config(page_title="DCF Intrinsic Value Calculator")

st.sidebar.title("Inputs")
ticker = st.sidebar.text_input("Enter stock ticker:", value="MCD").upper()
perpetualGrowthRate = st.sidebar.slider("Perpetual Growth Rate:", min_value=0.005, max_value=0.050, value=0.025, step=0.005, format="%.3f")

st.title("DCF Intrinsic Value Calculator")

if ticker:
    try:
        stock = yf.Ticker(ticker)

        wacc = get_WACC(stock)
        requiredReturnStart = wacc
        # if the WACC is unreasonable, set the required return to 7.5%
        if wacc > 0.12 or wacc < 0.05:
            requiredReturnStart = 0.075

        requiredReturn = st.sidebar.slider("Required Return:", min_value=0.01, max_value=0.20, value=requiredReturnStart,
                                                step=0.005, format="%.3f")
        st.sidebar.info("The default Required Return is the stock's WACC, if it's in the range 5%-12%.")

        fairValue, freeCashFlow, projectedFreeCashFlow, pastRevenue, projectedRevenue, pastNetIncome, projectedNetIncome = dcf(stock, requiredReturn, perpetualGrowthRate)

        currentPrice = stock.info.get("currentPrice")

        if currentPrice:
            # Determine colors based on fair value vs. current price
            if fairValue > currentPrice:
                boxColor = "#28a745"  # green
                textColor = "white"
            else:
                boxColor = "#dc3545"  # red
                textColor = "black"

            # Display the Fair Value prominently with background color and dynamic text color
            st.markdown(
                f"""
                <div style="background-color: {boxColor}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style='color:{textColor}'>{ticker}'s Fair Value: {round(fairValue, 2)}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Adding padding
            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style="border: 1px solid Gainsboro; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                    <h8><b>Current Price:</b> {round(currentPrice, 2)}</h8>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: Gainsboro; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style='color: black'>{ticker}'s Fair Value: {round(fairValue, 2)}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # Adding padding
            st.markdown("<br>", unsafe_allow_html=True)


        peg = stock.info.get("pegRatio")
        if peg:
            st.markdown(
                f"""
                <div style="border: 1px solid Gainsboro; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                    <h8><b>PEG Ratio:</b> {round(peg, 2)}</h8>
                </div>
                """,
                unsafe_allow_html=True,
            )

        pe = stock.info.get("trailingPE")
        if pe:
            st.markdown(
                f"""
                <div style="border: 1px solid Gainsboro; border-radius: 5px; text-align: center; margin-bottom: 10px;">
                    <h8><b>PE Ratio:</b> {round(pe, 2)}</h8>
                </div>
                """,
                unsafe_allow_html=True,
            )

        ev_ebitda = stock.info.get("enterpriseToEbitda")
        if ev_ebitda:
            st.markdown(
                f"""
                <div style="border: 1px solid Gainsboro; border-radius: 5px; text-align: center;">
                    <h8><b>EV/EBITDA:</b> {round(ev_ebitda, 2)}</h8>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.write("**Projected Free Cash Flow:**")
        overallCashFlow = pd.concat([projectedFreeCashFlow[1:], freeCashFlow])
        st.area_chart(overallCashFlow, color="#CECECE")

        st.write("**Projected Total Revenue:**")
        overallRevenue = pd.concat([projectedRevenue, pastRevenue])
        st.area_chart(overallRevenue, color="#4540DB")#"#060270")

        st.write("**Projected Net Income:**")
        overallIncome = pd.concat([projectedNetIncome, pastNetIncome])
        st.area_chart(overallIncome, color="#45D377")


    except KeyError as e:
        st.error(f"The Yahoo Finance API isn't providing the {e} data. Please provide a different stock, or try again later.")
