from xbk.Generate_model import generate_report

# 示例文本
test_up=""""Dow Soars 500 Points as Tech Giants Report Record Profits Ahead of Earnings Season.Fed Announces Unexpected Rate Cut, Sending Wall Street into Bullish Rally.Apple Unveils Revolutionary New iPhone, Sending Tech Stocks Skyrocketing.Oil Prices Plunge 10% Amid OPEC Supply Deal Collapse, Boosting Consumer Sentiment.Healthcare Sector Surges After FDA Approves Breakthrough Cancer Drug.Tesla Stock Jumps 20% on Rumors of New China Manufacturing Deal.
Wall Street Celebrates as Trade War With China Suddenly Halts, Tariffs Rolled Back.Goldman Sachs Reports Triple Quarterly Earnings, Sparking Financial Sector Rally.U.S. Unemployment Drops to 3.5%, Igniting Optimism in Retail and Consumer Stocks.Amazon Announces $10 Billion Green Energy Investment, Green Tech Stocks Surge.Pharma Giant Pfizer Secures Global Vaccine Patent, Shares Leap 15%.Bitcoin Surges Past $100K Amid Institutional Adoption Surge, Crypto Stocks Follow.
Boeing Receives $15 Billion Defense Contract, Aerospace Sector Soars.Retailers Report Holiday Sales Boom, Wal-Mart and Target Shares Hit Record Highs.Federal Infrastructure Bill Passes Congress, Boosting Construction and Steel Stocks.European Central Bank Signals Support for Dollar, Strengthening U.S. Exporters.FAANG Stocks Rally 8% as Analysts Upgrade Growth Forecasts for 2024.Tesla Opens World’s Largest Battery Plant, Shares Climb to Uncharted Heights.
Consumer Confidence Index Hits 15-Year High, Driving Auto and Travel Stocks.
Oil Majors Announce $50 Billion Dividend Boosts, Energy Sector Roars Back.Microsoft Acquires AI Startup for $20 Billion, Tech Sector Cheers Innovation.FDA Approves First Alzheimer’s Drug, Biotech Stocks Rally Across the Board.Fed Chair Signals Pause in Rate Hikes, Bond Yields Drop and Stocks Soar.Apple Pay Launches in India, Sending Global Payment Stocks into Overdrive.U.S. GDP Growth Revised Upward to 4%, Marking Strongest Quarter Since 2008"""

test_down="""The Dow plunges 800 points as tech giants report disappointing earnings ahead of season. The Fed signals a rate hike surprise, triggering a market sell-off. Apple faces a data breach crisis, shares collapse 12%. Oil prices surge 15% after OPEC+ cuts supply, consumer stocks plunge. The FDA halts clinical trials for a leading cancer drug, biotech sector tanks. Tesla warns of production delays, stock drops 25% on supply chain woes. 
Trade war escalates as China imposes new tariffs, global markets reel. Goldman Sachs reports a $2 billion loss, financial stocks dive. U.S. unemployment jumps to 6.5%, retail stocks plunge on weak consumer data. Amazon faces an antitrust lawsuit, shares drop 18% amid regulatory crackdown. Pfizer loses a patent battle, shares plunge as generic competitors enter the market. Bitcoin plummets 30% after a major exchange hack, crypto stocks crash.
Boeing delays 787 Dreamliner delivery, aerospace stocks collapse. Retailers report a holiday sales slump, Wal-Mart and Target shares hit 52-week lows. The infrastructure bill fails in Congress, construction stocks plunge. The ECB raises rates, weakening the dollar and harming U.S. exports. FAANG stocks crash 15% as analysts slash growth forecasts. Tesla shuts a Gigafactory over safety issues, shares plunge to an 18-month low.
Consumer confidence drops to a 10-year low, auto stocks collapse. Oil majors cut dividends amid price volatility, the energy sector tanks. 
Microsoft loses a major cloud contract, tech stocks slide. The FDA rejects an Alzheimer’s drug, the biotech sector plunges 10%. The Fed chair signals more rate hikes, bond yields spike and stocks plunge. Apple Pay faces a regulatory ban in India, payment stocks plummet. U.S. GDP growth is revised down to 1.2%, the worst quarter since 2009."""

# 分别处理两个示例文本
print("处理 test_up 的结果:")
output_up = generate_report(test_up)
print(output_up)

print("\n处理 test_down 的结果:")
output_down = generate_report(test_down)
print(output_down)

