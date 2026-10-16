import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv("data/demand_daily.csv", parse_dates=["date"])
x = d[(d.sku_id == "SKU02") & (d.date.dt.year == 2024)]
plt.plot(x.date, x.demand)
plt.title("Demande journalière SKU02 (lait), 2024")
plt.show()