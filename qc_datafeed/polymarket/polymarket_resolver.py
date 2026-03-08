import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import core.config as config


def get_today_btc_market():
    """
    解析当日活跃的 Polymarket "Bitcoin Up or Down" 市场。
    返回市场元信息，以及关键的两个时间锚点:
      - settlement_time: 该市场的结算时刻 (timezone-aware, ET)
      - k_reference_time: 用于获取 K 值的参考时刻 (timezone-aware, ET)
    """
    ny_tz = ZoneInfo(config.NY_TIMEZONE)
    ny_now = datetime.now(ny_tz)

    # Polymarket 结算规则: 每天美东 12:00 PM 划断
    # 如果当前已过 12:00 → 当天市场已结算/正在结算 → 活跃的是 "明天" 的市场
    # 如果当前未到 12:00 → 活跃的是 "今天" 的市场
    if ny_now.hour >= config.RESOLUTION_HOUR_ET:
        target_date = (ny_now + timedelta(days=1)).date()
    else:
        target_date = ny_now.date()

    # 该市场的结算时刻: target_date 的美东中午 12:00
    settlement_time = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        config.RESOLUTION_HOUR_ET,
        0,
        0,
        tzinfo=ny_tz,
    )

    # K 值参考时刻: 前一天的美东中午 12:00
    # 例: "March 8 Up or Down" 比较的是 Mar 7 noon vs Mar 8 noon
    k_ref_date = target_date - timedelta(days=1)
    k_reference_time = datetime(
        k_ref_date.year,
        k_ref_date.month,
        k_ref_date.day,
        config.RESOLUTION_HOUR_ET,
        0,
        0,
        tzinfo=ny_tz,
    )

    # 构造 slug
    month_name = target_date.strftime("%B").lower()
    day = target_date.day
    slug = f"{config.POLYMARKET_SLUG_PREFIX}{month_name}-{day}"

    url = f"https://gamma-api.polymarket.com/events/slug/{slug}"
    try:
        response = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"[Error] 网络请求失败: {e}")
        return None

    if response.status_code != 200:
        print(f"[Error] 未能找到市场 (HTTP {response.status_code}): {slug}")
        return None

    data = response.json()
    if "markets" not in data or not data["markets"]:
        print(f"[Error] 市场数据为空: {slug}")
        return None

    market = data["markets"][0]
    return {
        "slug": slug,
        "title": data.get("title"),
        "condition_id": market.get("conditionId"),
        "question": market.get("question"),
        "outcomes": market.get("outcomes", []),
        "settlement_time": settlement_time,
        "k_reference_time": k_reference_time,
        "gamma_market_raw": market,  # 原始 Gamma 数据, 供 CLOB 模块获取聚合价格
    }


if __name__ == "__main__":
    market = get_today_btc_market()
    if market:
        print(f"市场: {market['title']}")
        print(f"Condition ID: {market['condition_id']}")
        print(f"结算时间: {market['settlement_time']}")
        print(f"K 参考时间: {market['k_reference_time']}")
