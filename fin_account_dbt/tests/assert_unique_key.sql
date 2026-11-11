select slsfrc_id, efective_dt, count(*) as cnt
from {{ ref('slv_account') }}
group by 1, 2
having count(*) > 1
