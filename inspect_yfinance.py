import inspect
import yfinance as yf

print('yf.Tickers', hasattr(yf, 'Tickers'))
if hasattr(yf, 'Tickers'):
    print('signature', inspect.signature(yf.Tickers))
    print('Tickers attrs:')
    for name in sorted([name for name in dir(yf.Tickers) if not name.startswith('_')]):
        print(name)
print('Ticker info available', hasattr(yf.Ticker('AAPL'), 'info'))
print('Ticker fast_info available', hasattr(yf.Ticker('AAPL'), 'fast_info'))
