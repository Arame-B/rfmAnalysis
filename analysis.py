import pandas as pd


df = pd.read_csv("rfm_data.csv", parse_dates=["invoice_date"])

snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)




rfm = df.groupby("customer_id").agg({
    "invoice_date": lambda x: (snapshot_date - x.max()).days,  # Recency
    "invoice_id": "count",                                     # Frequency
    "amount": "sum"                                            # Monetary
}).reset_index()

rfm.rename(columns={
    "invoice_date": "Recency",
    "invoice_id": "Frequency",
    "amount": "Monetary"
}, inplace=True)

rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first", ascending=True), 5, labels=[5,4,3,2,1]).astype(int)
rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first", ascending=False), 5, labels=[5,4,3,2,1]).astype(int)
rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first", ascending=False), 5, labels=[5,4,3,2,1]).astype(int)

rfm["RFM Score"] = rfm[["R","F","M"]].sum(axis=1)

def classify_customer(score):
    if score >= 13:
        return "Best Customer"
    elif score >= 10:
        return "Loyal Customer"
    elif score >= 7:
        return "Needs Attention"
    else:
        return "At Risk"

rfm["Type customer"] = rfm["RFM Score"].apply(classify_customer)
rfm.to_csv("rfm_result.csv", index=False, encoding="utf-8-sig")





print(rfm)
