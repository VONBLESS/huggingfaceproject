# send_prices.py
import time
import datetime as dt
import serial
import yfinance as yf

PORT = "COM5"      # <-- your Arduino port
BAUD = 115200
UPDATE_SECS = 60

GOLD_SYMS  = ["XAUUSD=X", "GC=F"]     # currency spot, then COMEX gold futures
SILVER_SYMS= ["XAGUSD=X", "SI=F"]     # currency spot, then COMEX silver futures
USDINR_SYMS= ["USDINR=X", "INR=X"]    # both usually work for USD/INR

def last_price(sym):
    t = yf.Ticker(sym)
    # 1) fast_info
    try:
        p = t.fast_info.get("last_price", None)
        if p: return float(p)
    except Exception:
        pass
    # 2) 1m bars (today)
    try:
        h = t.history(period="1d", interval="1m")
        if not h.empty:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass
    # 3) daily close fallback
    try:
        h = t.history(period="5d")
        if not h.empty:
            return float(h["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None

def first_working(symbols):
    for s in symbols:
        p = last_price(s)
        if p is not None:
            return p, s
    return None, None

def format_lines():
    xau, xau_sym = first_working(GOLD_SYMS)
    xag, xag_sym = first_working(SILVER_SYMS)
    fx,  fx_sym  = first_working(USDINR_SYMS)

    if None in (xau, xag, fx):
        now = dt.datetime.now()
        l1 = "Au--.-k Ag--k"
        l2 = f"D:-- Upd {now:%H:%M}"
        return l1[:16], l2[:16]

    # Convert USD/oz -> INR per 10g / per kg
    OZ_TO_G = 31.1034768
    gold_inr_10g   = xau * fx * (10.0 / OZ_TO_G)
    silver_inr_kg  = xag * fx * (1000.0 / OZ_TO_G)

    # Compact to fit 16 chars
    # Example: "Au58.3k Ag75k"
    gold_k   = gold_inr_10g / 1000.0
    silver_k = silver_inr_kg / 1000.0

    l1 = f"Au{gold_k:.1f}k Ag{silver_k:.0f}k"
    # Show which feed we used (G=GC, S=SI) + time (optional)
    tag_g = "G" if xau_sym == "GC=F" else "g"
    tag_s = "S" if xag_sym == "SI=F" else "s"
    now = dt.datetime.now()
    l2 = f"{tag_g}{tag_s} {now:%H:%M}  D:--"

    return l1[:16], l2[:16]

def main():
    while True:
        try:
            with serial.Serial(PORT, BAUD, timeout=2) as ser:
                time.sleep(2)  # Arduino auto-reset grace
                while True:
                    l1, l2 = format_lines()
                    ser.write(f"L1:{l1}\n".encode("ascii", "ignore"))
                    ser.write(f"L2:{l2}\n".encode("ascii", "ignore"))
                    print(l1, "|", l2)
                    time.sleep(UPDATE_SECS)
        except serial.SerialException as e:
            print("Serial error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()
