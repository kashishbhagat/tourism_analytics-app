-- ===========================================================================
-- Tourism Experience Analytics -- analytical queries
--
-- These run against the SQLite database built by src/database.py
-- (data/processed/tourism.db), which holds the cleaned dimension tables plus
-- the integrated `master` table.
--
-- Each query is named with a `-- name:` tag; src/database.py parses these tags
-- so queries can be run individually by name from Python or the Streamlit app.
-- ===========================================================================


-- name: top_attractions
-- Highest-rated attractions with enough reviews to be trustworthy.
-- The HAVING clause is the point: a 5.0 average from two visits is noise, and
-- ranking on it would put flukes at the top of the leaderboard.
SELECT
    a.Attraction,
    a.AttractionType,
    a.AttractionCity,
    COUNT(*)                        AS visits,
    ROUND(AVG(a.Rating), 3)         AS mean_rating,
    SUM(CASE WHEN a.Rating >= 4 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS pct_satisfied
FROM master a
GROUP BY a.Attraction, a.AttractionType, a.AttractionCity
HAVING COUNT(*) >= 30
ORDER BY mean_rating DESC, visits DESC
LIMIT 20;


-- name: underperforming_attractions
-- The inverse view: high traffic, low satisfaction. This is the operational
-- priority list -- these attractions already have an audience, so fixing the
-- experience has immediate reach.
SELECT
    Attraction,
    AttractionType,
    AttractionCity,
    COUNT(*)                 AS visits,
    ROUND(AVG(Rating), 3)    AS mean_rating
FROM master
GROUP BY Attraction, AttractionType, AttractionCity
HAVING COUNT(*) >= 50
   AND AVG(Rating) < (SELECT AVG(Rating) FROM master)
ORDER BY visits DESC
LIMIT 20;


-- name: region_performance
-- Regional scorecard: volume, satisfaction and how far each region sits from
-- the global mean.
SELECT
    Continent,
    Region,
    COUNT(*)                                              AS visits,
    COUNT(DISTINCT UserId)                                AS users,
    ROUND(AVG(Rating), 3)                                 AS mean_rating,
    ROUND(AVG(Rating) - (SELECT AVG(Rating) FROM master), 3) AS vs_global
FROM master
GROUP BY Continent, Region
ORDER BY visits DESC;


-- name: visit_mode_by_continent
-- Visit mode mix per continent, as a share of that continent's visits.
-- Feeds the "who travels how, and where from" segmentation.
SELECT
    Continent,
    VisitMode,
    COUNT(*) AS visits,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY Continent), 2) AS pct_of_continent,
    ROUND(AVG(Rating), 3) AS mean_rating
FROM master
GROUP BY Continent, VisitMode
ORDER BY Continent, visits DESC;


-- name: seasonality
-- Monthly demand and satisfaction, with each month indexed against the
-- yearly average so the seasonal shape is readable independent of growth.
WITH monthly AS (
    SELECT
        VisitYear,
        VisitMonth,
        COUNT(*)      AS visits,
        AVG(Rating)   AS mean_rating
    FROM master
    GROUP BY VisitYear, VisitMonth
)
SELECT
    VisitMonth,
    SUM(visits)                                   AS total_visits,
    ROUND(AVG(mean_rating), 3)                    AS mean_rating,
    ROUND(SUM(visits) * 1.0 /
          (SELECT SUM(visits) * 1.0 / 12 FROM monthly), 3) AS seasonality_index
FROM monthly
GROUP BY VisitMonth
ORDER BY VisitMonth;


-- name: user_segments
-- Segment users by engagement and generosity. The CASE ladder turns two
-- continuous measures into the four segments the marketing team can act on.
WITH user_stats AS (
    SELECT
        UserId,
        Continent,
        Country,
        COUNT(*)     AS visit_count,
        AVG(Rating)  AS mean_rating
    FROM master
    GROUP BY UserId, Continent, Country
)
SELECT
    CASE
        WHEN visit_count >= 10 AND mean_rating >= 4 THEN 'Loyal advocates'
        WHEN visit_count >= 10 AND mean_rating <  4 THEN 'Frequent but unsatisfied'
        WHEN visit_count <  10 AND mean_rating >= 4 THEN 'Occasional advocates'
        ELSE                                             'At risk'
    END                          AS segment,
    COUNT(*)                     AS users,
    ROUND(AVG(visit_count), 2)   AS avg_visits,
    ROUND(AVG(mean_rating), 3)   AS avg_rating
FROM user_stats
GROUP BY segment
ORDER BY users DESC;


-- name: attraction_type_quadrant
-- Demand vs satisfaction quadrants per attraction type -- the SQL counterpart
-- of the scatter plot in the EDA report.
WITH t AS (
    SELECT
        AttractionType,
        COUNT(*)    AS visits,
        AVG(Rating) AS mean_rating
    FROM master
    GROUP BY AttractionType
),
benchmarks AS (
    SELECT AVG(visits) AS avg_visits, AVG(mean_rating) AS avg_rating FROM t
)
SELECT
    t.AttractionType,
    t.visits,
    ROUND(t.mean_rating, 3) AS mean_rating,
    CASE
        WHEN t.visits >= b.avg_visits AND t.mean_rating >= b.avg_rating THEN 'Star'
        WHEN t.visits >= b.avg_visits AND t.mean_rating <  b.avg_rating THEN 'Fix first'
        WHEN t.visits <  b.avg_visits AND t.mean_rating >= b.avg_rating THEN 'Promote'
        ELSE 'Deprioritise'
    END AS quadrant
FROM t CROSS JOIN benchmarks b
ORDER BY t.visits DESC;


-- name: repeat_visitor_rate
-- Retention proxy: what share of users return in a later year, by continent.
WITH user_years AS (
    SELECT UserId, Continent, COUNT(DISTINCT VisitYear) AS active_years
    FROM master
    GROUP BY UserId, Continent
)
SELECT
    Continent,
    COUNT(*)                                                       AS users,
    SUM(CASE WHEN active_years > 1 THEN 1 ELSE 0 END)              AS returning_users,
    ROUND(SUM(CASE WHEN active_years > 1 THEN 1 ELSE 0 END) * 100.0
          / COUNT(*), 2)                                           AS retention_pct
FROM user_years
GROUP BY Continent
ORDER BY retention_pct DESC;


-- name: cross_type_affinity
-- Which attraction types are visited by the same people? Drives
-- "customers who enjoyed X also enjoyed Y" bundling. The a.AttractionType <
-- b.AttractionType predicate keeps each unordered pair once and removes
-- self-pairs.
SELECT
    a.AttractionType AS type_a,
    b.AttractionType AS type_b,
    COUNT(DISTINCT a.UserId) AS shared_users
FROM master a
JOIN master b
  ON a.UserId = b.UserId
 AND a.AttractionType < b.AttractionType
GROUP BY a.AttractionType, b.AttractionType
ORDER BY shared_users DESC
LIMIT 15;


-- name: data_quality_check
-- Post-cleaning assertions. Every count here should be zero; if not, the
-- cleaning stage regressed.
SELECT 'ratings out of range'     AS check_name,
       COUNT(*)                   AS violations
FROM master WHERE Rating < 1 OR Rating > 5
UNION ALL
SELECT 'months out of range', COUNT(*) FROM master WHERE VisitMonth < 1 OR VisitMonth > 12
UNION ALL
SELECT 'null ratings',        COUNT(*) FROM master WHERE Rating IS NULL
UNION ALL
SELECT 'null visit modes',    COUNT(*) FROM master WHERE VisitMode IS NULL
UNION ALL
SELECT 'duplicate transactions',
       (SELECT COUNT(*) FROM (SELECT TransactionId FROM master
                              GROUP BY TransactionId HAVING COUNT(*) > 1));
