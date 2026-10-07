from jevlab.latency_study import SYNTH_BANDS, build_items


def test_items_disjoint_from_test_and_fit_jstep_cap():
    dev = [{"id": f"jev-d-{i:03d}", "problem": f"p{i}"} for i in range(1, 21)]
    items = build_items(dev, 20, 3)
    assert len(items) == 20 + 3 * len(SYNTH_BANDS)
    assert not any(it["item_id"].startswith("jev-t-") for it in items)
    assert len({it["item_id"] for it in items}) == len(items)
    # n + FINAL line must fit the 64-round JSTEP cap
    assert max(SYNTH_BANDS.values()) + 2 + 1 <= 64
