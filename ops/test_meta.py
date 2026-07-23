import bittensor as bt
st = bt.Subtensor(network="finney")
mg = st.metagraph(netuid=81)
print("metagraph n =", mg.n)
w = bt.Wallet(name="camille81-v2", hotkey="hotkey81")
hk = w.hotkey.ss58_address
print("hotkey81 =", hk)
if hk in mg.hotkeys:
    uid = mg.hotkeys.index(hk)
    print(f"ENREGISTREE uid={uid} incentive={mg.I[uid]:.6f} active={mg.active[uid]}")
else:
    print("*** NON ENREGISTREE sur netuid 81 ***")
