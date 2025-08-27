# src/data.py
import argparse, os, requests, pandas as pd

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)

DROP_COLS = ["ID"]
CAT_CANDIDATES = ["SEX", "EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in range(1, 7)]

def download_uci(to_dir: str) -> str:
    os.makedirs(to_dir, exist_ok=True)
    out_path = os.path.join(to_dir, "default_of_credit_card_clients.xls")
    r = requests.get(UCI_URL, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f: f.write(r.content)
    print(f"Downloaded: {out_path}")
    return out_path

def make_clean_csv(excel_path: str, out_csv: str):
    df = pd.read_excel(excel_path, header=1)
    if "PAY_0" in df.columns and "PAY_1" not in df.columns:
        df = df.rename(columns={"PAY_0": "PAY_1"})
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} shape={df.shape}")

def load_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    y = df["default"].astype(int).values
    X = df.drop(columns=["default"])
    cat_cols = [c for c in CAT_CANDIDATES if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, y, num_cols, cat_cols

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true")
    p.add_argument("--to", default="data/raw")
    p.add_argument("--make-clean-csv", default=None)
    p.add_argument("--excel", default="data/raw/default_of_credit_card_clients.xls")
    args = p.parse_args()
    if args.download:
        download_uci(args.to)
    if args.make_clean_csv:
        make_clean_csv(args.excel, args.make_clean_csv)
