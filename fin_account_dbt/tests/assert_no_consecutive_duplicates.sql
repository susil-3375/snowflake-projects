select slsfrc_id, efective_dt
from {{ ref('slv_account') }}
qualify row_hash = lag(row_hash) over (partition by slsfrc_id order by efective_dt)
