"""Small, explicitly synthetic reports for a reproducible first run."""

DEMO_COMPANY = "海川制造（虚构示例）"


def demo_reports() -> list[tuple[str, int, bytes]]:
    reports = []
    for year, revenue, cash, profit, explanation in (
        (
            2022,
            "100",
            "18",
            "12",
            "公司客户回款保持稳定，产品销售与经营现金流同步增长。",
        ),
        (
            2023,
            "120",
            "15",
            "14",
            "公司扩大信用销售，应收账款增加，客户回款周期延长，经营现金流承压。",
        ),
        (
            2024,
            "150",
            "9",
            "16",
            "营业收入增长主要来自新产品交付。应收账款与存货增加，回款滞后于收入确认，经营现金流下降。",
        ),
    ):
        text = (
            f"# {DEMO_COMPANY} {year} 年度报告\n"
            "本文件为人工构造的教学示例，不是真实上市公司公告，不可用作投资依据。\n\n"
            "## 主要财务指标\n单位：亿元\n"
            f"{year} 年营业收入为 {revenue} 亿元。\n"
            f"{year} 年经营活动产生的现金流量净额为 {cash} 亿元。\n"
            f"{year} 年归属于上市公司股东的净利润为 {profit} 亿元。\n"
            "\f"
            "## 管理层讨论与分析\n"
            f"{explanation}\n\n"
            "## 风险提示\n"
            "公司面临客户集中、应收账款回收及原材料价格波动风险。以上均为合成案例。\n"
        )
        reports.append((f"海川制造_{year}_合成示例.md", year, text.encode("utf-8")))
    return reports
