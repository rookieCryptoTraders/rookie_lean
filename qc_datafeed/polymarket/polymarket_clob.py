import requests
import json
import core.config as config


def get_polymarket_prices(condition_id, outcome="Up", gamma_market=None):
    """
    获取 Polymarket 指定方向的完整价格信息。

    Polymarket 价格来源层级:
      - Gamma events/slug API: 包含 bestAsk, bestBid, outcomePrices, 最准确的聚合数据
      - CLOB /markets: 包含 token.price (指示价)
      - CLOB /book: rest limit orders (通常只有边缘挂单, 不含 AMM 流动性)

    参数:
      condition_id: market conditionId
      outcome: "Up" 或 "Down"
      gamma_market: 可选, 来自 events/slug 的 market 原始数据 (避免重复请求)
    """
    result = {
        "best_ask": None,
        "best_bid": None,
        "last_trade_price": None,
        "clob_best_ask": None,
        "is_liquid": False,
    }

    # ── 1. 从 Gamma events/slug 获取真实聚合盘口 ──
    if gamma_market:
        outcomes_raw = gamma_market.get("outcomes", "[]")
        prices_raw = gamma_market.get("outcomePrices", "[]")

        try:
            outcomes_list = (
                json.loads(outcomes_raw)
                if isinstance(outcomes_raw, str)
                else outcomes_raw
            )
            prices_list = (
                json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            )
        except (json.JSONDecodeError, TypeError):
            outcomes_list = []
            prices_list = []

        # 找到对应 outcome 的价格
        for i, oc in enumerate(outcomes_list):
            if oc == outcome and i < len(prices_list):
                result["best_ask"] = float(prices_list[i])
                # 构建 bid: 使用 spread
                spread = float(gamma_market.get("spread", 0.01) or 0.01)
                result["best_bid"] = max(0.01, result["best_ask"] - spread)
                break

        ltp = gamma_market.get("lastTradePrice")
        result["last_trade_price"] = float(ltp) if ltp else None

    # ── 2. 从 CLOB /markets 获取 token 指示价作为补充 ──
    try:
        clob_url = f"https://clob.polymarket.com/markets/{condition_id}"
        clob_res = requests.get(clob_url, timeout=10).json()
        tokens = clob_res.get("tokens", [])
        target_token = None
        for token in tokens:
            if token.get("outcome") == outcome:
                target_token = token
                break

        if target_token:
            token_id = target_token.get("token_id")
            clob_price = float(target_token.get("price", 0) or 0)

            # 如果 Gamma 没返回价格, 用 CLOB token.price 作为 fallback
            if result["best_ask"] is None and clob_price > 0:
                result["best_ask"] = clob_price
                result["best_bid"] = max(0.01, clob_price - 0.01)

            # 获取 CLOB 限价挂单簿 (参考)
            if token_id:
                try:
                    book_url = f"https://clob.polymarket.com/book?token_id={token_id}"
                    book_res = requests.get(book_url, timeout=10).json()
                    asks = book_res.get("asks", [])
                    if asks:
                        result["clob_best_ask"] = float(asks[0]["price"])
                except requests.exceptions.RequestException:
                    pass

    except requests.exceptions.RequestException as e:
        print(f"[CLOB] 网络请求失败: {e}")

    # ── 3. 判断流动性 ──
    if result["best_ask"] is not None:
        result["is_liquid"] = result["best_ask"] < config.THIN_BOOK_ASK_THRESHOLD

    return result


if __name__ == "__main__":
    # 测试: 用 events/slug 获取 gamma_market 再传入
    slug = "bitcoin-up-or-down-on-march-8"
    r = requests.get(
        f"https://gamma-api.polymarket.com/events/slug/{slug}", timeout=10
    ).json()
    gamma_market = r["markets"][0]
    cond = gamma_market["conditionId"]
    print(f"condition_id: {cond}")
    for side in ["Up", "Down"]:
        prices = get_polymarket_prices(cond, side, gamma_market)
        print(
            f"[{side}] ask={prices['best_ask']} bid={prices['best_bid']} "
            f"last={prices['last_trade_price']} clob_ask={prices['clob_best_ask']} "
            f"liquid={prices['is_liquid']}"
        )
