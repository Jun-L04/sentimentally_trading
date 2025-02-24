# sentimentally_trading
Fine-tuned OPT model trading sentimentally with many emotions, 'replicating' arXiv:[2412.19245]

Limitations:
- Finnhub only offers one year of historical news lookback for free tier, in order to test the model, we only trained the model on some montsh of 2024 and tested it on the remaining months
- Finnhub API is buggy, ex, it did not return anything for the month of janurary and feburary 
- Pandas next avaialbe business day BDay() for some reason does not recognize independence day and good friday