{% set attr_cols = ['account_name','account_type','industry','annual_revenue','account_status','billing_state'] %}

select
    trim(slsfrc_id)               as slsfrc_id,
    efective_dt,
    trim(account_name)            as account_name,
    initcap(trim(account_type))   as account_type,
    trim(industry)                as industry,
    annual_revenue,
    initcap(trim(account_status)) as account_status,
    upper(trim(billing_state))    as billing_state,
    md5(concat_ws('|'
        {%- for c in attr_cols %}, coalesce({{ c }}::varchar, '~'){% endfor %}
    ))                            as row_hash,
    ingested_ts
from {{ source('raw', 'vw_account_src') }}
where slsfrc_id is not null and efective_dt is not null
-- one row per natural key; latest ingestion wins on reloads
qualify row_number() over (partition by slsfrc_id, efective_dt order by ingested_ts desc) = 1
