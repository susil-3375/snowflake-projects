{{
  config(
    materialized         = 'incremental',
    incremental_strategy = 'merge',
    unique_key           = ['slsfrc_id', 'efective_dt'],
    on_schema_change     = 'fail',
    cluster_by           = ['slsfrc_id']
  )
}}

{% set cols = 'slsfrc_id, efective_dt, account_name, account_type, industry, annual_revenue, account_status, billing_state, row_hash, ingested_ts' %}

with src as (
    -- CDC filter: only rows ingested since the last load
    select {{ cols }}, 'SRC' as rec_src
    from {{ ref('stg_account') }}
    {% if is_incremental() %}
    where ingested_ts > (select coalesce(max(ingested_ts), '1900-01-01'::timestamp_ntz) from {{ this }})
    {% endif %}
)

{% if is_incremental() %}
, tgt_latest as (
    -- latest Silver row per affected account = comparison baseline
    select {{ cols }}, 'TGT' as rec_src
    from {{ this }}
    where slsfrc_id in (select slsfrc_id from src)
    qualify row_number() over (partition by slsfrc_id order by efective_dt desc) = 1
)
{% endif %}

, unioned as (
    select * from src
    {% if is_incremental() %}
    union all
    select * from tgt_latest
    {% endif %}
)

, deduped as (
    -- a key present in both (reload) keeps the existing Silver row
    select * from unioned
    qualify row_number() over (
        partition by slsfrc_id, efective_dt order by iff(rec_src = 'TGT', 0, 1)
    ) = 1
)

, compared as (
    select *, lag(row_hash) over (partition by slsfrc_id order by efective_dt) as prev_hash
    from deduped
)

select
    slsfrc_id, efective_dt, account_name, account_type, industry,
    annual_revenue, account_status, billing_state, row_hash, ingested_ts,
    current_timestamp()::timestamp_ntz as dbt_inserted_ts
from compared
where rec_src = 'SRC'
  and (prev_hash is null or row_hash <> prev_hash)
