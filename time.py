import time
from web3 import Web3

# 1. Preprocessing time
t0 = time.time()
master = preprocess_uploaded(bene_df, inp_df, outp_df)
print(f"Preprocessing: {time.time() - t0:.2f}s")

# 2. Prediction time
t1 = time.time()
proba = lr_model.predict_proba(X_scaled)[:, 1]
print(f"Prediction: {(time.time() - t1)*1000:.1f}ms")

# 3. Blockchain — already in your receipt logs
# Print this after each storeFraudRecord() call:
print(f"Gas Used: {receipt['gasUsed']}")
print(f"Block: {receipt['blockNumber']}")