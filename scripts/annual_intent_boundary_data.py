"""Matched finance-intent examples for training and confidence calibration."""

# Each pair preserves the metric and target year while changing one condition.
TRAIN_PAIRS = (
    ("lookup", "{year}年度{metric}合计有多少钱", "{year}年度{metric}的直营部分合计有多少钱"),
    ("lookup", "{metric}在{year}年实际是多少亿", "{metric}在{year}年最后两个月实际是多少亿"),
    ("lookup", "{company}{year}年{metric}大概多大", "{company}{year}年单体报表里的{metric}大概多大"),
    ("lookup", "{year}年{metric}的总数能给我吗", "{year}年把新合并主体去掉以后的{metric}总数能给我吗"),
    ("lookup", "想查一下{year}年{metric}有多少万元", "想查一下{year}年{metric}有多少万元以及哪块业务占得最多"),
    ("lookup", "告诉我{metric}在{year}年度的具体金额", "告诉我{metric}在{year}年度抵销内部交易前的具体金额"),
    ("yoy", "{year}年{metric}跟上一年比较少了多少钱", "{year}年{metric}跟上上年比较少了多少钱"),
    ("yoy", "{metric}在{year}年跟上一财年比高了几个点", "{metric}在{year}年跟上一财年比高了几个点，主要是什么在驱动"),
    ("yoy", "较上年同期，{year}年{metric}多出多少钱", "较上年同期，{year}年单月{metric}多出多少钱"),
    ("yoy", "{year}年{metric}比上年度多多少万元", "{year}年{metric}比上年度多多少万元，有多少可以归因于提价"),
    ("yoy", "{company}{year}年度{metric}和上年相差多少", "{company}{year}年度{metric}和竞争对手上年相差多少"),
    ("yoy", "{year}年{metric}较上个财年的增幅是多少", "{year}年{metric}按重列前数据算，较上个财年的增幅是多少"),
)

CALIBRATION_PAIRS = (
    ("lookup", "{year}年度{metric}到底达多少万元", "{year}年度{metric}的联营业务贡献到底达多少万元"),
    ("lookup", "说一下{metric}在{year}年实际是多少钱", "说一下{metric}在{year}年其中四至六月实际是多少钱"),
    ("lookup", "{company}{year}年{metric}的具体总数是多少", "{company}{year}年附注按地区拆分的{metric}总数是多少"),
    ("yoy", "{year}年{metric}与上一年度相比少了几个点", "{year}年{metric}在汇率不变口径下与上一年度相比少了几个点"),
    ("yoy", "跟去年同期对比，{year}年{metric}多出多少亿", "跟去年同期对比，{year}年{metric}多出多少亿，能否延续"),
    ("yoy", "{metric}于{year}年度比上个财年少了多少", "{metric}于{year}年度比行业上个财年的水平少了多少"),
)

# These examples share the question ending even when the requested scope changes.
HARD_TRAIN_PAIRS = (
    ("lookup", "{year}年{metric}的实际数字是多少", "{year}年剔除一次性事项后的{metric}实际数字是多少"),
    ("lookup", "{metric}在{year}年一共是多少", "{metric}在{year}年只计可比业务一共是多少"),
    ("lookup", "{company}{year}年{metric}到底多少钱", "{company}{year}年{metric}按调整前口径到底多少钱"),
    ("lookup", "能告诉我{year}年度{metric}总额吗", "能告诉我{year}年度仅限本部的{metric}总额吗"),
    ("lookup", "{year}年的{metric}合计多少钱呢", "{year}年只含主业的{metric}合计多少钱呢"),
    ("lookup", "{metric}于{year}年实际有多少亿元", "{metric}于{year}年用附注中的重分类口径实际有多少亿元"),
    ("yoy", "跟上一年度比，{year}年{metric}净增多少钱", "跟历史最低值比，{year}年{metric}净增多少钱"),
    ("yoy", "{year}年{metric}与上年同期比多多少", "{year}年{metric}与三年平均相比多多少"),
    ("yoy", "{company}{year}年度{metric}和上一个财年差多少", "{company}{year}年度{metric}和一家同业的上一个财年差多少"),
    ("yoy", "{year}年{metric}和去年同期相比减少几亿", "{year}年{metric}按固定汇率和去年同期相比减少几亿"),
    ("yoy", "和上个年度比，{year}年{metric}多了多少钱", "和上个年度比，{year}年{metric}中由新业务带来的部分多了多少钱"),
    ("yoy", "{metric}在{year}年度与去年比变化多少", "{metric}在{year}年度与去年比变化多少由会计估计变更引起"),
)

HARD_SELECTION_PAIRS = (
    ("lookup", "{year}年度{metric}实际金额有多少", "{year}年度{metric}按调整后的可比范围实际金额有多少"),
    ("lookup", "{company}的{metric}在{year}年合计是多少", "{company}的{metric}在{year}年仅由控股子公司贡献的部分合计是多少"),
    ("lookup", "告诉我{year}年{metric}大约多少万元", "告诉我{year}年{metric}中由会计政策变更导致的金额大约多少万元"),
    ("yoy", "{year}年{metric}和上一年相比多出几亿", "{year}年{metric}和上市首年相比多出几亿"),
    ("yoy", "{company}{year}年{metric}对照去年同期涨了几个点", "{company}{year}年{metric}对照同行去年同期涨了几个点"),
    ("yoy", "较去年同期，{year}年{metric}多出来多少钱", "较去年同期，{year}年{metric}扣掉汇兑影响以后多出来多少钱"),
)


def pair_templates(pairs, label):
    if label == "open":
        return tuple(boundary for _, _, boundary in pairs)
    return tuple(simple for operation, simple, _ in pairs if operation == label)
