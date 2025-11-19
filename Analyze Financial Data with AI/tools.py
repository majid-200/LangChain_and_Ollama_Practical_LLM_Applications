"""
Financial Data Tools Module
============================
This module provides LangChain tools for fetching financial data:
1. Historical stock prices (via Yahoo Finance)
2. SEC filing sections (via EDGAR database)

These tools are designed to be used by AI agents/LLMs to gather financial
information during analysis or decision-making tasks.
"""

from datetime import datetime, timedelta
from enum import Enum

import yfinance as yf  # Yahoo Finance API for stock data
from edgar import Company  # SEC EDGAR database for company filings
from langchain_core.tools import tool  # Decorator to create LangChain-compatible tools

# RESPONSE TEMPLATES

# Purpose: Define XML-structured output formats for consistency and parseability
#
# Why XML format?
# - Structured: Easy for LLMs and code to parse
# - Explicit tags: Clear boundaries between data elements
# - Hierarchical: Can nest complex data relationships

# Template for stock price data
# Example output:
#   <prices>
#   <2024-01-07> 150.25</2024-01-07>
#   <2024-01-14> 152.80</2024-01-14>
#   </prices>
PRICES_RESPONSE_TEMPLATE = """
<prices>
{prices}
</prices>
""".strip()  # .strip() removes leading/trailing whitespace

# Template for SEC filing data
# Example output:
#   <filing>
#       <company>NVIDIA Corporation</company>
#       <filing_date>2024-01-15</filing_date>
#       <sections>...</sections>
#   </filing>
COMPANY_FILING_RESPONE_TEMPLATE = """
<filing>
    <company>{company}</company>
    <filing_date>{filing_date}</filing_date>
    <sections>{sections}</sections>
</filing>
""".strip()

# TOOL 1: HISTORICAL STOCK PRICE FETCHER

# Purpose: Retrieve weekly stock closing prices for the last 90 days
#
# Visual Flow:
#   Input (ticker) → Calculate dates → Fetch daily data → Resample to weekly → Format XML
#       ↓                  ↓                  ↓                   ↓               ↓
#     "NVDA"         90 days back      yfinance API         Group by week    <prices>...</prices>
#
# Why weekly instead of daily?
# - Reduces noise: Daily fluctuations can be erratic
# - Manageable data: ~13 weeks vs 90 days of data
# - Better for trends: Easier to spot patterns

@tool  # ← Decorator makes this function a LangChain tool
       # LangChain tools can be called by AI agents automatically
       # The @tool decorator reads the docstring to understand what the function does
def get_historical_stock_price(ticker: str) -> str:
    """
    Fetches weekly historical stock price data for a given company ticker for the last 90 days.

    Args:
        ticker: The stock ticker symbol (e.g., "NVDA", "AMD").

    Returns:
        A string with lines of the date and weekly close price in the last 90 days.
    """
    try:
        # STEP 1: Calculate Date Range (Last 90 Days)

        # Timeline visualization:
        #   [90 days ago] ←──────────────────→ [Today]
        #   start_date                         end_date
        
        end_date = datetime.now()  # Get current date/time
        start_date = end_date - timedelta(days=90)  # Subtract 90 days
        
        # Convert to string format required by yfinance API
        # Format: "YYYY-MM-DD" (e.g., "2024-01-15")
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # STEP 2: Fetch Daily Stock Data from Yahoo Finance

        # yfinance returns a DataFrame with columns:
        #   Date | Open | High | Low | Close | Volume | Dividends | Stock Splits
        
        stock = yf.Ticker(ticker)  # Create ticker object (e.g., yf.Ticker("NVDA"))

        # Fetch daily historical data
        daily_history = stock.history(
            start=start_date_str, 
            end=end_date_str, 
            interval="1d"  # "1d" = daily data points
        )

        # STEP 3: Resample Daily Data to Weekly

        # Resampling visualization:
        #   Daily:  Mon Tue Wed Thu Fri | Mon Tue Wed Thu Fri | ...
        #   Weekly:         [Week 1]            [Week 2]
        #                  (Sunday)           (Sunday)
        #
        # "W-SUN" = Weekly, ending on Sunday
        # .last() = Take the last value of each week (Friday's close usually)
        
        weekly_resampled = daily_history["Close"].resample("W-SUN").last()
        # Result: Series with weekly closing prices indexed by Sunday dates

        # STEP 4: Format Data into XML Structure
        
        lines = []  # List to collect formatted price lines

        # Iterate through dates and prices simultaneously
        for date, close in zip(weekly_resampled.index, weekly_resampled):
            # Remove timezone info for cleaner formatting
            date = date.replace(tzinfo=None)
            
            # Ensure date doesn't exceed today (edge case handling)
            if date > end_date:
                date = end_date
            
            # Format date as string: "2024-01-15"
            date_text = date.strftime("%Y-%m-%d")
            
            # Create XML-style line: <2024-01-15>150.25</2024-01-15>
            # round(close, 2) = Round to 2 decimal places (e.g., 150.246 → 150.25)
            lines.append(f"<{date_text}>{round(close, 2)}</{date_text}>")

        # STEP 5: Return Formatted XML Response

        # Join all lines with newlines and insert into template
        return PRICES_RESPONSE_TEMPLATE.format(prices="\n".join(lines))
    
    except Exception as e:
        # If anything fails (bad ticker, network error, etc.), return error message
        return f"An error occurred while fetching historical price for '{ticker}': {str(e)}"
    
# SEC FILING SECTIONS ENUM

# Purpose: Define valid SEC filing sections that can be retrieved
#
# Visual Structure:
#   Section (Enum)
#   ├── MDA                  → Management Discussion & Analysis
#   ├── RISK_FACTORS         → Company risk disclosures
#   ├── BALANCE_SHEET        → Assets, Liabilities, Equity snapshot
#   ├── INCOME_STATEMENT     → Revenue, Expenses, Profit/Loss
#   └── CASHFLOW_STATEMENT   → Cash inflows/outflows
#
# SEC Form 10-Q Structure:
#   ┌─────────────────────────────────────┐
#   │ Part I - Financial Information      │
#   │   Item 1: Financial Statements      │ ← balance_sheet, income, cashflow
#   │   Item 2: MD&A                      │ ← mda
#   │                                     │
#   │ Part II - Other Information         │
#   │   Item 1A: Risk Factors             │ ← risk_factors
#   └─────────────────────────────────────┘
    
class Section(str, Enum):
    """
    Enumeration of SEC filing sections that can be fetched.
    
    Why use Enum?
    - Type safety: Prevents requesting invalid sections
    - Clear options: IDE autocomplete shows available sections
    - Consistent naming: "mda" not "MD&A" or "management_discussion"
    """
    MDA = "mda"  # Management's Discussion & Analysis
                 # - Management's perspective on financial performance
                 # - Explains trends, risks, and future outlook
                 
    RISK_FACTORS = "risk_factors"  # Risk Factors disclosure
                                    # - Legal, operational, market risks
                                    # - Required by SEC for transparency
                                    
    BALANCE_SHEET = "balance_sheet"  # Statement of Financial Position
                                      # - Assets = Liabilities + Equity
                                      # - Snapshot at a point in time
                                      
    INCOME_STATEMENT = "income_statement"  # Statement of Operations
                                           # - Revenue - Expenses = Net Income
                                           # - Shows profitability over time
                                           
    CASHFLOW_STATEMENT = "cashflow_statement"  # Statement of Cash Flows
                                               # - Operating, Investing, Financing activities
                                               # - Tracks actual cash movement

# TOOL 2: SEC FILING SECTION FETCHER

# Purpose: Retrieve specific sections from a company's most recent 10-Q filing
#
# Visual Flow:
#   Input → Get Company → Find Latest 10-Q → Parse XBRL → Extract Sections → Format XML
#     ↓          ↓             ↓                 ↓              ↓              ↓
#   NVDA    Company obj    Filing obj      Statements obj   MD&A, etc    <filing>...</filing>
#
# What is a 10-Q?
# - Quarterly report filed with SEC
# - Contains financial statements + management commentary
# - Required for all public companies (3 times/year, not Q4)
#
# What is XBRL?
# - eXtensible Business Reporting Language
# - Structured format for financial data
# - Makes financial statements machine-readable

@tool  # ← LangChain tool decorator (same as above)
def fetch_sec_filing_sections(
    ticker: str,
    sections: list[Section],  # List of Section enum values to retrieve
    ) -> str:
    """
    Fetches specific sections from company's last SEC filing.

    Args:
        ticker: The ticker symbol of the company to fetch the SEC filing for.
        sections: The sections to fetch from the SEC filing. Available sections are (mda, risk_factors, balance_sheet, income_statement, cashflow_statement).

    Returns:
        A string with the company's SEC filing sections in XML format.
    """
    
    # STEP 1: Initialize Company and Get Latest 10-Q Filing

    # EDGAR database hierarchy:
    #   Company → Filings (many) → Latest Filing → Sections
    
    company = Company(ticker)  # Create company object from ticker
    
    # Get all 10-Q filings for this company
    # Note: 10-Q = Quarterly report (10-K would be annual report)
    filings = company.get_filings(form="10-Q")
    
    # Get the most recent filing
    filing = filings.latest()
    
    # STEP 2: Parse Filing Data

    # Two types of data in SEC filings:
    # 1. Textual sections (MD&A, Risk Factors) → filing.obj()
    # 2. Financial statements (numbers) → filing.xbrl()
    
    filing_obj = filing.obj()  # Get parsed textual content
                                # Contains "Item 1", "Item 2", etc.
                                
    xbrl = filing.xbrl()  # Get XBRL data (structured financial statements)
    statements = xbrl.statements  # Access financial statement objects

    # STEP 3: Extract Requested Sections

    # Map each Section enum to its data source
    
    sections_data = {}  # Dictionary to store: {Section: data}
    
    for section in sections:
        if section == Section.MDA:
            # Item 2 = Management's Discussion & Analysis
            sections_data[section] = filing_obj["Item 2"]
            
        elif section == Section.RISK_FACTORS:
            # Item 1A = Risk Factors
            sections_data[section] = filing_obj["Item 1A"]
            
        elif section == Section.BALANCE_SHEET:
            # Call method to extract balance sheet from XBRL
            sections_data[section] = statements.balance_sheet()
            
        elif section == Section.INCOME_STATEMENT:
            # Call method to extract income statement from XBRL
            sections_data[section] = statements.income_statement()
            
        elif section == Section.CASHFLOW_STATEMENT:
            # Call method to extract cash flow statement from XBRL
            sections_data[section] = statements.cashflow_statement()

    # STEP 4: Format into XML Response

    # Build nested XML structure:
    #   <filing>
    #     <company>NVIDIA</company>
    #     <filing_date>2024-01-15</filing_date>
    #     <sections>
    #       <mda>... content ...</mda>
    #       <balance_sheet>... content ...</balance_sheet>
    #     </sections>
    #   </filing>
    
    return COMPANY_FILING_RESPONE_TEMPLATE.format(
        company=company.name,  # Company name (e.g., "NVIDIA Corporation")
        filing_date=filing.filing_date.strftime("%Y-%m-%d"),  # Format date
        sections="\n".join(  # Join all section XML strings with newlines
            [
                # For each section, wrap data in tags: <section_name>data</section_name>
                f"<({section.value})>\n{data}\n</{section.value}>"
                for section, data in sections_data.items()
            ]
        ),
    )


# USAGE EXAMPLES
"""
How to use these tools in your code:

# ───────────────────────────────────────────────────────────────────────────
# Example 1: Fetch Historical Stock Prices
# ───────────────────────────────────────────────────────────────────────────
from tools import get_historical_stock_price

# Get NVIDIA's weekly stock prices for last 90 days
nvda_prices = get_historical_stock_price.invoke({"ticker": "NVDA"})
print(nvda_prices)
# Output:
# <prices>
# <2024-01-07>495.20</2024-01-07>
# <2024-01-14>530.45</2024-01-14>
# ...
# </prices>

# ───────────────────────────────────────────────────────────────────────────
# Example 2: Fetch SEC Filing Sections
# ───────────────────────────────────────────────────────────────────────────
from tools import fetch_sec_filing_sections, Section

# Get MD&A and Risk Factors from AMD's latest 10-Q
amd_filing = fetch_sec_filing_sections.invoke({
    "ticker": "AMD",
    "sections": [Section.MDA, Section.RISK_FACTORS]
})
print(amd_filing)

# Get all financial statements
all_financials = fetch_sec_filing_sections.invoke({
    "ticker": "AAPL",
    "sections": [
        Section.BALANCE_SHEET,
        Section.INCOME_STATEMENT,
        Section.CASHFLOW_STATEMENT
    ]
})

# ───────────────────────────────────────────────────────────────────────────
# Example 3: Using with LangChain Agents
# ───────────────────────────────────────────────────────────────────────────
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from tools import get_historical_stock_price, fetch_sec_filing_sections

# Create list of available tools for the agent
tools = [get_historical_stock_price, fetch_sec_filing_sections]

# Agent can now automatically call these tools when analyzing:
# "What were NVIDIA's stock trends last quarter and what risks do they mention?"
"""