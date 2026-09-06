"""Script to align and configure all 8 platform conversion goals in Matomo.
"""

import requests
import json
import time

import os

MATOMO_URL = os.getenv("MATOMO_URL", "https://data.fpldna.com/matomo/index.php")
TOKEN = os.getenv("MATOMO_TOKEN", "")
SITE_ID = int(os.getenv("MATOMO_SITE_ID", "3"))

DESIRED_GOALS = [
    {
        "id": 1,
        "name": "Onboarding Completed",
        "description": "User finished institutional orientation tour",
        "allowMultiple": 0,
    },
    {
        "id": 2,
        "name": "Pre-Flight Trade Cleared",
        "description": "User validated 5/5 trade sanity checks",
        "allowMultiple": 1,
    },
    {
        "id": 3,
        "name": "Trade Plan Copied",
        "description": "User copied Markdown execution plan for trade journal",
        "allowMultiple": 1,
    },
    {
        "id": 4,
        "name": "Portfolio Position Added",
        "description": "User logged trade into private browser storage with position value revenue",
        "allowMultiple": 1,
    },
    {
        "id": 5,
        "name": "Price Alert Created",
        "description": "User configured breakout pivot or pullback alert",
        "allowMultiple": 1,
    },
    {
        "id": 6,
        "name": "Macro Stress Simulation Run",
        "description": "User ran beta-weighted market crash stress test",
        "allowMultiple": 1,
    },
    {
        "id": 7,
        "name": "Stock Comparison Run",
        "description": "User executed head-to-head competitor factor analysis",
        "allowMultiple": 1,
    },
    {
        "id": 8,
        "name": "Screener Filter Applied",
        "description": "User filtered Peter Lynch/Magic Formula/GARP gems",
        "allowMultiple": 1,
    },
]

def main():
    print(f"Connecting to Matomo at {MATOMO_URL} for Site ID: {SITE_ID}...")
    headers = {"Authorization": f"Bearer {TOKEN}"}

    # 1. Fetch current goals
    res = requests.post(
        MATOMO_URL,
        data={
            "module": "API",
            "method": "Goals.getGoals",
            "idSite": SITE_ID,
            "format": "json",
            "token_auth": TOKEN,
        },
        headers=headers,
        timeout=10,
    )
    existing = res.json()
    print(f"Current goals in Matomo: {json.dumps(existing, indent=2)}")
    existing_by_id = {int(x["idgoal"]): x for x in existing if not x.get("deleted")}

    # 2. Configure Goals 1 through 8
    for g in DESIRED_GOALS:
        gid = g["id"]
        g_name = g["name"]

        if gid in existing_by_id:
            # Update existing goal
            print(f"[UPDATING] Updating Goal #{gid} -> '{g_name}'...")
            upd_res = requests.post(
                MATOMO_URL,
                data={
                    "module": "API",
                    "method": "Goals.updateGoal",
                    "idSite": SITE_ID,
                    "idGoal": gid,
                    "name": g_name,
                    "matchAttribute": "manually",
                    "pattern": "manual",
                    "patternType": "contains",
                    "allowMultiple": g["allowMultiple"],
                    "revenue": 0.0,
                    "format": "json",
                    "token_auth": TOKEN,
                },
                headers=headers,
                timeout=10,
            )
            print(f"  Result: {upd_res.text.strip()}")
        else:
            # Add new goal
            print(f"[CREATING] Adding Goal #{gid}: '{g_name}'...")
            add_res = requests.post(
                MATOMO_URL,
                data={
                    "module": "API",
                    "method": "Goals.addGoal",
                    "idSite": SITE_ID,
                    "name": g_name,
                    "matchAttribute": "manually",
                    "pattern": "manual",
                    "patternType": "contains",
                    "allowMultiple": g["allowMultiple"],
                    "revenue": 0.0,
                    "format": "json",
                    "token_auth": TOKEN,
                },
                headers=headers,
                timeout=10,
            )
            print(f"  Result: {add_res.text.strip()}")
        time.sleep(0.2)

    # 3. Final Verification
    verify_res = requests.post(
        MATOMO_URL,
        data={
            "module": "API",
            "method": "Goals.getGoals",
            "idSite": SITE_ID,
            "format": "json",
            "token_auth": TOKEN,
        },
        headers=headers,
        timeout=10,
    )
    final_goals = verify_res.json()
    print("\n" + "="*70)
    print(f"CONFIRMED: {len(final_goals)} Active Goals in Matomo for Site {SITE_ID} (Arx Terminal):")
    print("="*70)
    for fg in sorted(final_goals, key=lambda x: int(x.get("idgoal", 0))):
        print(f"  Goal #{fg.get('idgoal')}: {fg.get('name')} | Trigger: {fg.get('match_attribute')} | Multiple: {fg.get('allow_multiple')}")
    print("="*70)

if __name__ == "__main__":
    main()
