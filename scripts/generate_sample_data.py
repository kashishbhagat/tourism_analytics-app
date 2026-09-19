"""
Generate a schema-accurate sample of the Tourism dataset.

The real dataset is distributed as a Google Drive workbook. This script
produces the same nine tables with the same column names and dtypes so the
entire pipeline can be developed and tested without it. Drop the real files
into data/raw/ and everything downstream works unchanged.

Realistic defects are injected on purpose -- missing values, inconsistent
city-name casing and whitespace, duplicate transaction rows, out-of-range
ratings and a handful of orphan foreign keys -- so that the cleaning stage
has something to actually clean.

Usage:
    python scripts/generate_sample_data.py --users 3000 --transactions 40000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DIR, RANDOM_STATE  # noqa: E402

# --------------------------------------------------------------------------
# Reference data
# --------------------------------------------------------------------------
CONTINENTS = ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"]

REGIONS = {
    "Asia": ["South East Asia", "South Asia", "East Asia", "Western Asia"],
    "Europe": ["Western Europe", "Southern Europe", "Northern Europe", "Eastern Europe"],
    "North America": ["Northern America", "Central America", "Caribbean"],
    "South America": ["South America"],
    "Africa": ["East Africa", "West Africa", "Southern Africa", "North Africa"],
    "Oceania": ["Australia and New Zealand", "Melanesia"],
}

COUNTRIES = {
    "South East Asia": ["Indonesia", "Malaysia", "Singapore", "Thailand", "Vietnam", "Philippines"],
    "South Asia": ["India", "Sri Lanka", "Nepal", "Bangladesh", "Pakistan"],
    "East Asia": ["China", "Japan", "South Korea", "Taiwan"],
    "Western Asia": ["United Arab Emirates", "Saudi Arabia", "Israel", "Turkey"],
    "Western Europe": ["France", "Germany", "Netherlands", "Belgium"],
    "Southern Europe": ["Italy", "Spain", "Greece", "Portugal"],
    "Northern Europe": ["United Kingdom", "Sweden", "Norway", "Denmark"],
    "Eastern Europe": ["Poland", "Czech Republic", "Hungary", "Romania"],
    "Northern America": ["United States", "Canada", "Mexico"],
    "Central America": ["Costa Rica", "Panama", "Guatemala"],
    "Caribbean": ["Jamaica", "Cuba", "Dominican Republic"],
    "South America": ["Brazil", "Argentina", "Peru", "Chile", "Colombia"],
    "East Africa": ["Kenya", "Tanzania", "Uganda", "Ethiopia"],
    "West Africa": ["Nigeria", "Ghana", "Senegal"],
    "Southern Africa": ["South Africa", "Botswana", "Namibia"],
    "North Africa": ["Egypt", "Morocco", "Tunisia"],
    "Australia and New Zealand": ["Australia", "New Zealand"],
    "Melanesia": ["Fiji", "Papua New Guinea"],
}

CITY_SEEDS = [
    "Jakarta", "Bandung", "Surabaya", "Kuala Lumpur", "Penang", "Singapore",
    "Bangkok", "Phuket", "Hanoi", "Manila", "Mumbai", "Delhi", "Bengaluru",
    "Chennai", "Colombo", "Kathmandu", "Dhaka", "Lahore", "Shanghai",
    "Beijing", "Tokyo", "Osaka", "Seoul", "Taipei", "Dubai", "Riyadh",
    "Tel Aviv", "Istanbul", "Paris", "Lyon", "Berlin", "Munich", "Amsterdam",
    "Brussels", "Rome", "Milan", "Madrid", "Barcelona", "Athens", "Lisbon",
    "London", "Manchester", "Stockholm", "Oslo", "Copenhagen", "Warsaw",
    "Prague", "Budapest", "Bucharest", "New York", "Los Angeles", "Chicago",
    "Toronto", "Vancouver", "Mexico City", "San Jose", "Panama City",
    "Guatemala City", "Kingston", "Havana", "Santo Domingo", "Sao Paulo",
    "Rio de Janeiro", "Buenos Aires", "Lima", "Santiago", "Bogota",
    "Nairobi", "Dar es Salaam", "Kampala", "Addis Ababa", "Lagos", "Accra",
    "Dakar", "Cape Town", "Johannesburg", "Gaborone", "Windhoek", "Cairo",
    "Marrakesh", "Tunis", "Sydney", "Melbourne", "Auckland", "Wellington",
    "Suva", "Port Moresby",
]

ATTRACTION_TYPES = [
    "Beaches", "Ancient Ruins", "History Museums", "Religious Sites",
    "Nature & Wildlife Areas", "Water Parks", "Volcanos", "Caverns & Caves",
    "Spas", "National Parks", "Points of Interest & Landmarks",
    "Neighborhoods", "Flea & Street Markets", "Waterfalls", "Ballets",
]

VISIT_MODES = ["Business", "Couples", "Family", "Friends", "Solo"]

ATTRACTION_NAME_PARTS = {
    "Beaches": ["Beach", "Bay", "Cove", "Shore"],
    "Ancient Ruins": ["Temple Ruins", "Ancient City", "Archaeological Park"],
    "History Museums": ["Museum", "Heritage Centre", "Historical Museum"],
    "Religious Sites": ["Temple", "Cathedral", "Mosque", "Shrine"],
    "Nature & Wildlife Areas": ["Wildlife Reserve", "Nature Park", "Sanctuary"],
    "Water Parks": ["Water Park", "Aqua Park", "Splash World"],
    "Volcanos": ["Volcano", "Crater", "Mount"],
    "Caverns & Caves": ["Caves", "Cavern", "Grotto"],
    "Spas": ["Hot Springs", "Thermal Spa", "Wellness Retreat"],
    "National Parks": ["National Park", "Reserve", "Conservation Area"],
    "Points of Interest & Landmarks": ["Tower", "Monument", "Landmark", "Bridge"],
    "Neighborhoods": ["Old Town", "Quarter", "District"],
    "Flea & Street Markets": ["Night Market", "Bazaar", "Street Market"],
    "Waterfalls": ["Falls", "Cascades", "Waterfall"],
    "Ballets": ["Opera House", "Ballet Theatre", "Performing Arts Centre"],
}


def build_geography(rng: np.random.Generator):
    """Build the continent / region / country / city lookup tables."""
    continent_df = pd.DataFrame(
        {"ContinentId": range(1, len(CONTINENTS) + 1), "Continent": CONTINENTS}
    )
    cont_id = dict(zip(continent_df.Continent, continent_df.ContinentId))

    region_rows = []
    for cont, regions in REGIONS.items():
        for reg in regions:
            region_rows.append({"Region": reg, "ContinentId": cont_id[cont]})
    region_df = pd.DataFrame(region_rows)
    region_df.insert(0, "RegionId", range(1, len(region_df) + 1))
    reg_id = dict(zip(region_df.Region, region_df.RegionId))

    country_rows = []
    for reg, countries in COUNTRIES.items():
        for c in countries:
            country_rows.append({"Country": c, "RegionId": reg_id[reg]})
    country_df = pd.DataFrame(country_rows)
    country_df.insert(0, "CountryId", range(1, len(country_df) + 1))

    # Spread cities across countries, keeping a plausible 1-4 cities each.
    city_rows = []
    pool = list(CITY_SEEDS)
    rng.shuffle(pool)
    idx = 0
    for country_id in country_df.CountryId:
        n = int(rng.integers(1, 4))
        for _ in range(n):
            name = pool[idx % len(pool)]
            idx += 1
            city_rows.append({"CityName": name, "CountryId": int(country_id)})
    city_df = pd.DataFrame(city_rows)
    city_df.insert(0, "CityId", range(1, len(city_df) + 1))

    return continent_df, region_df, country_df, city_df


def build_attractions(rng: np.random.Generator, city_df: pd.DataFrame, n: int):
    type_df = pd.DataFrame(
        {
            "AttractionTypeId": range(1, len(ATTRACTION_TYPES) + 1),
            "AttractionType": ATTRACTION_TYPES,
        }
    )
    rows = []
    for i in range(1, n + 1):
        t_id = int(rng.integers(1, len(ATTRACTION_TYPES) + 1))
        t_name = ATTRACTION_TYPES[t_id - 1]
        city = city_df.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        suffix = ATTRACTION_NAME_PARTS[t_name][
            int(rng.integers(0, len(ATTRACTION_NAME_PARTS[t_name])))
        ]
        rows.append(
            {
                "AttractionId": i,
                "AttractionCityId": int(city.CityId),
                "AttractionTypeId": t_id,
                "Attraction": f"{city.CityName} {suffix}",
                "AttractionAddress": f"{int(rng.integers(1, 400))} Main Road, {city.CityName}",
            }
        )
    return type_df, pd.DataFrame(rows)


def build_transactions(
    rng: np.random.Generator,
    user_df: pd.DataFrame,
    item_df: pd.DataFrame,
    type_df: pd.DataFrame,
    n_txn: int,
):
    """
    Generate ratings with genuine structure so the models have real signal:
      * every attraction has a latent quality score
      * every user has a latent generosity bias
      * each visit mode shifts the rating slightly (business travellers rate lower)
      * users have a preferred attraction type and rate it higher
    """
    n_users = len(user_df)
    n_items = len(item_df)

    item_quality = rng.normal(3.9, 0.55, n_items)
    user_bias = rng.normal(0.0, 0.40, n_users)
    # Popularity is heavy-tailed: a few attractions dominate the visits.
    item_pop = rng.pareto(1.4, n_items) + 1
    item_pop = item_pop / item_pop.sum()
    user_pop = rng.pareto(1.6, n_users) + 1
    user_pop = user_pop / user_pop.sum()

    mode_effect = {"Business": -0.35, "Couples": 0.25, "Family": 0.05,
                   "Friends": 0.15, "Solo": -0.05}
    # Mode probabilities vary by continent -- gives the classifier real signal.
    base_mode_p = np.array([0.12, 0.26, 0.30, 0.22, 0.10])

    type_of_item = dict(zip(item_df.AttractionId, item_df.AttractionTypeId))
    n_types = len(type_df)
    user_pref_type = rng.integers(1, n_types + 1, n_users)

    user_ids = user_df.UserId.to_numpy()
    item_ids = item_df.AttractionId.to_numpy()

    u_idx = rng.choice(n_users, size=n_txn, p=user_pop)

    # Item choice is part popularity, part personal taste. A purely
    # popularity-driven draw would leave no personal signal in *which*
    # attraction a user visits, and then a popularity ranking is by
    # construction the optimal recommender -- which is not how real tourism
    # data behaves. PREF_SHARE of visits are drawn from the user's preferred
    # attraction type instead.
    PREF_SHARE = 0.55
    item_type_arr = item_df.AttractionTypeId.to_numpy()
    items_by_type = {
        t: np.where(item_type_arr == t)[0] for t in np.unique(item_type_arr)
    }
    # Popularity restricted to each type, renormalised.
    pop_by_type = {
        t: item_pop[idxs] / item_pop[idxs].sum() for t, idxs in items_by_type.items()
    }

    i_idx = np.empty(n_txn, dtype=int)
    use_pref = rng.random(n_txn) < PREF_SHARE
    for k in range(n_txn):
        if use_pref[k]:
            t = user_pref_type[u_idx[k]]
            pool, probs = items_by_type[t], pop_by_type[t]
            i_idx[k] = int(rng.choice(pool, p=probs))
        else:
            i_idx[k] = int(rng.choice(n_items, p=item_pop))

    continent_ids = user_df.ContinentId.to_numpy()

    rows = []
    for k in range(n_txn):
        ui, ii = int(u_idx[k]), int(i_idx[k])
        cont = int(continent_ids[ui])

        p = base_mode_p.copy()
        p[0] += 0.06 if cont in (1, 2) else -0.02   # more business travel
        p[2] += 0.07 if cont in (3, 4) else -0.01   # more family travel
        p = np.clip(p, 0.02, None)
        p = p / p.sum()
        mode = VISIT_MODES[int(rng.choice(len(VISIT_MODES), p=p))]

        year = int(rng.choice([2020, 2021, 2022, 2023, 2024],
                              p=[0.10, 0.15, 0.22, 0.26, 0.27]))
        month = int(rng.integers(1, 13))
        # Peak season lifts ratings a little.
        season_lift = 0.12 if month in (6, 7, 8, 12) else 0.0

        pref_lift = 0.45 if type_of_item[item_ids[ii]] == user_pref_type[ui] else 0.0

        raw = (
            item_quality[ii]
            + user_bias[ui]
            + mode_effect[mode]
            + season_lift
            + pref_lift
            + rng.normal(0, 0.55)
        )
        rating = int(np.clip(round(raw), 1, 5))

        rows.append(
            {
                "TransactionId": k + 1,
                "UserId": int(user_ids[ui]),
                "VisitYear": year,
                "VisitMonth": month,
                "VisitMode": mode,
                "AttractionId": int(item_ids[ii]),
                "Rating": rating,
            }
        )
    return pd.DataFrame(rows)


def inject_defects(rng: np.random.Generator, tables: dict[str, pd.DataFrame]):
    """Dirty the data the way a real export is dirty."""
    txn = tables["Transaction"]

    # 1. Duplicate rows (~0.5%)
    dupes = txn.sample(frac=0.005, random_state=RANDOM_STATE)
    txn = pd.concat([txn, dupes], ignore_index=True)

    # 2. Missing ratings (~1.5%) and missing visit modes (~0.8%)
    miss_r = rng.choice(txn.index, size=int(0.015 * len(txn)), replace=False)
    txn.loc[miss_r, "Rating"] = np.nan
    miss_m = rng.choice(txn.index, size=int(0.008 * len(txn)), replace=False)
    txn.loc[miss_m, "VisitMode"] = np.nan

    # 3. Out-of-range ratings and impossible months
    bad_r = rng.choice(txn.index, size=int(0.004 * len(txn)), replace=False)
    txn.loc[bad_r, "Rating"] = rng.choice([0, 6, 9, -1], size=len(bad_r))
    bad_m = rng.choice(txn.index, size=int(0.003 * len(txn)), replace=False)
    txn.loc[bad_m, "VisitMonth"] = rng.choice([0, 13, 14], size=len(bad_m))

    # 4. Orphan foreign keys
    orph = rng.choice(txn.index, size=int(0.002 * len(txn)), replace=False)
    txn.loc[orph, "AttractionId"] = 999_999

    # 5. Casing / whitespace / spelling noise in visit mode labels
    noisy = rng.choice(txn.index, size=int(0.03 * len(txn)), replace=False)
    txn.loc[noisy, "VisitMode"] = (
        txn.loc[noisy, "VisitMode"].astype("object").map(
            lambda v: v if not isinstance(v, str)
            else rng.choice([v.upper(), v.lower(), f"  {v} ", v])
        )
    )
    tables["Transaction"] = txn

    # 6. City names with inconsistent casing and stray whitespace
    city = tables["City"]
    noisy_c = rng.choice(city.index, size=max(1, int(0.12 * len(city))), replace=False)
    city.loc[noisy_c, "CityName"] = city.loc[noisy_c, "CityName"].map(
        lambda v: rng.choice([v.upper(), v.lower(), f" {v}  "])
    )
    tables["City"] = city

    # 7. A few users with a missing city
    user = tables["User"]
    miss_city = rng.choice(user.index, size=max(1, int(0.01 * len(user))), replace=False)
    user.loc[miss_city, "CityId"] = np.nan
    tables["User"] = user

    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--attractions", type=int, default=350)
    ap.add_argument("--transactions", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=RANDOM_STATE)
    ap.add_argument("--format", choices=["csv", "xlsx"], default="csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    continent_df, region_df, country_df, city_df = build_geography(rng)

    # Users live in a city; region/continent are derived from it so the
    # hierarchy is internally consistent.
    city_lookup = city_df.merge(country_df, on="CountryId").merge(region_df, on="RegionId")
    picks = city_lookup.sample(args.users, replace=True, random_state=args.seed)
    user_df = pd.DataFrame(
        {
            "UserId": range(1, args.users + 1),
            "ContinentId": picks.ContinentId.to_numpy(),
            "RegionId": picks.RegionId.to_numpy(),
            "CountryId": picks.CountryId.to_numpy(),
            "CityId": picks.CityId.to_numpy(),
        }
    )

    type_df, item_df = build_attractions(rng, city_df, args.attractions)
    txn_df = build_transactions(rng, user_df, item_df, type_df, args.transactions)

    mode_df = pd.DataFrame(
        {"VisitModeId": range(1, len(VISIT_MODES) + 1), "VisitMode": VISIT_MODES}
    )

    tables = {
        "Transaction": txn_df,
        "User": user_df,
        "City": city_df,
        "Type": type_df,
        "Mode": mode_df,
        "Continent": continent_df,
        "Country": country_df,
        "Region": region_df,
        "Item": item_df,
    }
    tables = inject_defects(rng, tables)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        if args.format == "csv":
            path = RAW_DIR / f"{name}.csv"
            df.to_csv(path, index=False)
        else:
            path = RAW_DIR / f"{name}.xlsx"
            df.to_excel(path, index=False)
        print(f"  wrote {path.name:20s} {len(df):>7,} rows x {df.shape[1]} cols")

    print(f"\nSample dataset written to {RAW_DIR}")


if __name__ == "__main__":
    main()
