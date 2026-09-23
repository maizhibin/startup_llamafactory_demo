#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a fully synthetic startup-incubation demo dataset.

This script intentionally uses fake people/projects/institutions. It demonstrates how
A-F business data can be transformed into task datasets and an LLaMA-Factory SFT set.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

SEED = 20260922
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

N_FOUNDERS = 80
N_PROJECTS = 80
TRAIN_PER_TASK = 30
EVAL_PER_TASK = 8

educations = ["大专", "本科", "硕士", "博士"]
majors = ["计算机", "机械工程", "电子信息", "材料科学", "工商管理", "生物工程", "设计", "金融"]
groups = ["高校毕业生", "技能人才", "科研人员", "企业技术骨干", "返乡创业者", "普通创业者"]
venture_types = ["科技创业", "制造业创业", "数字服务", "消费服务", "文化创意"]
industries = ["人工智能", "机器人", "智能制造", "新能源", "生物医药", "企业服务", "消费科技", "文化数字化"]
cities = ["杭州", "宁波", "南京", "苏州", "合肥", "武汉", "成都", "深圳"]
stages = ["机会探索", "项目筹备", "MVP验证", "市场验证", "商业化", "成长期"]
rounds = ["未融资", "天使轮", "Pre-A", "A轮"]
course_catalog = [
    ("C01", "客户发现与需求验证", "客户发现"),
    ("C02", "B2B销售实战", "销售"),
    ("C03", "创业财务与现金流", "财务"),
    ("C04", "产品经理与MVP设计", "产品"),
    ("C05", "团队管理与组织建设", "管理"),
    ("C06", "创业融资与路演", "融资"),
    ("C07", "知识产权与合规", "合规"),
    ("C08", "产业市场分析", "市场分析"),
]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, int(round(x))))

# -------------------- Raw A-F demo data --------------------
founders = []
employment = []
projects = []
training_records = []
finance_risk = []

for i in range(1, N_FOUNDERS + 1):
    fid = f"F{i:04d}"
    edu = random.choice(educations)
    major = random.choice(majors)
    group = random.choice(groups)
    vtype = random.choice(venture_types)
    city = random.choice(cities)
    years = random.randint(1, 18)
    reg = random.choices(["未登记", "已登记"], weights=[30, 70])[0]
    founders.append({
        "founder_id": fid,
        "display_name": f"创业者{i:04d}",
        "education": edu,
        "major": major,
        "group_type": group,
        "registration_status": reg,
        "venture_type": vtype,
        "years_experience": years,
        "city": city,
    })
    credit_band = random.choices(["A", "B", "C"], weights=[55, 35, 10])[0]
    employment.append({
        "founder_id": fid,
        "insured_status": random.choice(["正常", "创业后转灵活就业", "暂停"]),
        "contribution_years": round(random.uniform(0.5, 16), 1),
        "interruption_count": random.randint(0, 3),
        "credit_band": credit_band,
        "contract_default_count": 0 if credit_band == "A" else random.randint(0, 2),
    })

    stage = random.choices(stages, weights=[10, 12, 22, 24, 20, 12])[0]
    industry = random.choice(industries)
    team = random.randint(2, 28)
    patents = random.randint(0, 8)
    customers = random.randint(0, 60)
    revenue = random.choice([0, 0, 5, 10, 20, 50, 100, 200, 500]) * 10000
    fr = random.choice(rounds)
    pid = f"P{i:04d}"
    projects.append({
        "project_id": pid,
        "founder_id": fid,
        "project_name": f"示例项目{i:04d}",
        "industry": industry,
        "team_size": team,
        "stage": stage,
        "patent_count": patents,
        "registered": "是" if reg == "已登记" else random.choice(["是", "否"]),
        "operating_status": random.choices(["正常", "试运营", "暂停"], weights=[75, 20, 5])[0],
        "revenue_12m": revenue,
        "customer_count": customers,
        "funding_round": fr,
    })

    # 2-4 training records per founder
    for j in range(random.randint(2, 4)):
        cid, cname, skill = random.choice(course_catalog)
        pre = random.randint(35, 78)
        post = min(98, pre + random.randint(6, 24))
        training_records.append({
            "founder_id": fid,
            "course_id": cid,
            "course_name": cname,
            "skill_tag": skill,
            "completed": random.choices(["是", "否"], weights=[88, 12])[0],
            "pre_score": pre,
            "post_score": post,
            "instructor_qualification": random.choice(["行业导师", "高级讲师", "创业导师"]),
        })

    abnormal = random.choices([0, 1], weights=[88, 12])[0]
    legal = random.choices([0, 1, 2], weights=[85, 12, 3])[0]
    dishonest = random.choices([0, 1], weights=[97, 3])[0]
    market_growth = round(random.uniform(-0.05, 0.45), 2)
    financing_need = random.choice([0, 100, 300, 500, 800, 1200]) * 10000
    finance_risk.append({
        "project_id": pid,
        "funding_round": fr,
        "funding_amount": random.choice([0, 0, 100, 300, 500, 1000]) * 10000,
        "valuation": random.choice([0, 1000, 2000, 5000, 10000]) * 10000,
        "operating_abnormality": abnormal,
        "legal_cases": legal,
        "dishonesty_record": dishonest,
        "market_growth": market_growth,
        "financing_need": financing_need,
    })

policies = []
for i in range(1, 25):
    policies.append({
        "policy_id": f"POL{i:03d}",
        "policy_name": f"示例创业扶持政策{i:03d}",
        "region": random.choice(cities),
        "industries": "|".join(random.sample(industries, k=random.randint(1, 3))),
        "target_stages": "|".join(random.sample(stages, k=random.randint(1, 3))),
        "support_type": random.choice(["补贴", "创业担保贷款", "场地支持", "导师服务", "研发资助"]),
        "amount_max": random.choice([10, 20, 50, 100, 200]) * 10000,
        "valid_to": random.choice(["2026-12-31", "2027-06-30", "2027-12-31"]),
    })

investors = []
for i in range(1, 25):
    investors.append({
        "investor_id": f"INV{i:03d}",
        "investor_name": f"示例投资机构{i:03d}",
        "preferred_industries": "|".join(random.sample(industries, k=random.randint(2, 4))),
        "preferred_stages": "|".join(random.sample(["天使轮", "Pre-A", "A轮"], k=random.randint(1, 2))),
        "ticket_min": random.choice([100, 300, 500]) * 10000,
        "ticket_max": random.choice([800, 1200, 2000, 3000]) * 10000,
        "region": random.choice(cities + ["全国"]),
    })

market_signals = []
for industry in industries:
    for quarter in ["2025Q4", "2026Q1", "2026Q2"]:
        market_signals.append({
            "industry": industry,
            "quarter": quarter,
            "new_company_growth": round(random.uniform(-0.1, 0.5), 2),
            "funding_growth": round(random.uniform(-0.2, 0.7), 2),
            "patent_growth": round(random.uniform(-0.1, 0.5), 2),
            "job_growth": round(random.uniform(-0.1, 0.45), 2),
            "policy_strength": round(random.uniform(0.35, 0.95), 2),
            "market_growth": round(random.uniform(-0.05, 0.45), 2),
        })

write_csv(RAW / "A_founders.csv", founders)
write_csv(RAW / "B_employment_credit.csv", employment)
write_csv(RAW / "C_projects.csv", projects)
write_csv(RAW / "D_training_records.csv", training_records)
write_csv(RAW / "E_policies.csv", policies)
write_csv(RAW / "F_finance_risk.csv", finance_risk)
write_csv(RAW / "F_investors.csv", investors)
write_csv(RAW / "F_market_signals.csv", market_signals)

# Indexes
employment_by_f = {x["founder_id"]: x for x in employment}
project_by_f = {x["founder_id"]: x for x in projects}
finance_by_p = {x["project_id"]: x for x in finance_risk}
train_by_f = defaultdict(list)
for x in training_records:
    train_by_f[x["founder_id"]].append(x)
market_by_ind = defaultdict(list)
for x in market_signals:
    market_by_ind[x["industry"]].append(x)

# -------------------- Derived task datasets --------------------
founder_profile = []
venture_eval = []
skill_training = []
policy_match = []
investment_match = []
risk_opp = []

for f in founders:
    p = project_by_f[f["founder_id"]]
    e = employment_by_f[f["founder_id"]]
    trs = train_by_f[f["founder_id"]]
    skill_gain = round(sum(int(x["post_score"]) - int(x["pre_score"]) for x in trs) / max(1, len(trs)), 1)
    tech_base = 55 + (10 if f["major"] in ["计算机", "机械工程", "电子信息", "材料科学", "生物工程"] else 0) + min(15, f["years_experience"])
    commercial = 42 + min(20, int(p["customer_count"]) // 2) + (8 if int(p["revenue_12m"]) > 0 else 0)
    management = 40 + min(30, int(p["team_size"]) * 2)
    founder_profile.append({
        "founder_id": f["founder_id"],
        "snapshot_time": "2026-06-30",
        "education": f["education"],
        "major": f["major"],
        "group_type": f["group_type"],
        "venture_type": f["venture_type"],
        "years_experience": int(f["years_experience"]),
        "project_stage": p["stage"],
        "technical_capability": clamp(tech_base),
        "commercial_capability": clamp(commercial),
        "management_capability": clamp(management),
        "training_gain": skill_gain,
        "quality_score": 0.96,
        "label_source": "synthetic_rule_demo"
    })

    team_score = clamp(48 + int(p["team_size"]) * 1.5 + min(10, f["years_experience"]))
    tech_score = clamp(52 + int(p["patent_count"]) * 5 + (8 if f["major"] in ["计算机", "机械工程", "电子信息", "材料科学", "生物工程"] else 0))
    market_score = clamp(45 + int(p["customer_count"]) * 0.8 + finance_by_p[p["project_id"]]["market_growth"] * 40)
    traction_score = clamp(40 + min(35, int(p["customer_count"])) + (15 if int(p["revenue_12m"]) > 0 else 0))
    risk_penalty = finance_by_p[p["project_id"]]["operating_abnormality"] * 15 + finance_by_p[p["project_id"]]["legal_cases"] * 8 + finance_by_p[p["project_id"]]["dishonesty_record"] * 25
    overall = clamp(0.25*team_score + 0.25*tech_score + 0.25*market_score + 0.25*traction_score - risk_penalty)
    venture_eval.append({
        "project_id": p["project_id"], "founder_id": f["founder_id"], "snapshot_time": "2026-06-30",
        "industry": p["industry"], "stage": p["stage"], "team_size": int(p["team_size"]),
        "patent_count": int(p["patent_count"]), "customer_count": int(p["customer_count"]),
        "revenue_12m": int(p["revenue_12m"]), "team_score": team_score, "technology_score": tech_score,
        "market_score": market_score, "traction_score": traction_score, "overall_score": overall,
        "quality_score": 0.94, "label_source": "synthetic_rule_demo"
    })

    # one skill recommendation per founder based on weakest synthetic dimension
    dims = {"销售": commercial, "管理": management, "财务": 50 + min(20, skill_gain), "市场分析": market_score}
    weak_skill = min(dims, key=dims.get)
    candidates = [c for c in course_catalog if c[2] == weak_skill]
    if not candidates:
        candidates = course_catalog
    cid, cname, skill = candidates[0]
    skill_training.append({
        "founder_id": f["founder_id"], "project_stage": p["stage"], "weak_skill": weak_skill,
        "current_score": clamp(dims[weak_skill]), "course_id": cid, "course_name": cname,
        "match_label": 1, "expected_improvement": random.randint(8, 18), "label_source": "synthetic_rule_demo"
    })

    # policy pair: positive/negative by region + industry + stage
    policy = random.choice(policies)
    eligible = int(f["city"] == policy["region"] and p["industry"] in policy["industries"].split("|") and p["stage"] in policy["target_stages"].split("|"))
    policy_match.append({
        "project_id": p["project_id"], "policy_id": policy["policy_id"], "city": f["city"], "industry": p["industry"],
        "stage": p["stage"], "policy_region": policy["region"], "policy_industries": policy["industries"],
        "policy_target_stages": policy["target_stages"], "eligible": eligible,
        "match_score": 92 if eligible else random.randint(20, 68), "label_source": "synthetic_rule_demo"
    })

    inv = random.choice(investors)
    need = int(finance_by_p[p["project_id"]]["financing_need"])
    round_for_match = p["funding_round"] if p["funding_round"] != "未融资" else "天使轮"
    match = int(p["industry"] in inv["preferred_industries"].split("|") and round_for_match in inv["preferred_stages"].split("|") and (need == 0 or int(inv["ticket_min"]) <= need <= int(inv["ticket_max"])))
    investment_match.append({
        "project_id": p["project_id"], "investor_id": inv["investor_id"], "industry": p["industry"],
        "stage": round_for_match, "financing_need": need, "preferred_industries": inv["preferred_industries"],
        "preferred_stages": inv["preferred_stages"], "ticket_min": int(inv["ticket_min"]), "ticket_max": int(inv["ticket_max"]),
        "match_label": match, "match_score": 90 if match else random.randint(15, 70), "label_source": "synthetic_rule_demo"
    })

    fr = finance_by_p[p["project_id"]]
    latest_market = market_by_ind[p["industry"]][-1]
    risk_score = clamp(fr["operating_abnormality"]*35 + fr["legal_cases"]*18 + fr["dishonesty_record"]*45 + (20 if int(p["revenue_12m"]) == 0 and p["stage"] in ["商业化", "成长期"] else 0))
    opp_score = clamp(50 + latest_market["funding_growth"]*25 + latest_market["patent_growth"]*20 + latest_market["job_growth"]*20 + latest_market["policy_strength"]*15)
    risk_opp.append({
        "project_id": p["project_id"], "industry": p["industry"], "snapshot_time": "2026-06-30",
        "operating_abnormality": int(fr["operating_abnormality"]), "legal_cases": int(fr["legal_cases"]),
        "dishonesty_record": int(fr["dishonesty_record"]), "market_growth": float(fr["market_growth"]),
        "risk_score": risk_score, "risk_level": "高" if risk_score >= 60 else "中" if risk_score >= 30 else "低",
        "opportunity_score": opp_score, "opportunity_level": "高" if opp_score >= 70 else "中" if opp_score >= 50 else "低",
        "label_source": "synthetic_rule_demo"
    })

write_jsonl(PROC / "founder_profile.jsonl", founder_profile)
write_jsonl(PROC / "venture_stage_evaluation.jsonl", venture_eval)
write_jsonl(PROC / "skill_gap_training.jsonl", skill_training)
write_jsonl(PROC / "policy_resource_matching.jsonl", policy_match)
write_jsonl(PROC / "investment_matching.jsonl", investment_match)
write_jsonl(PROC / "risk_opportunity.jsonl", risk_opp)

# CPT demo text: deliberately small; for format smoke test only.
cpt_docs = []
for p in projects[:50]:
    f = next(x for x in founders if x["founder_id"] == p["founder_id"])
    cpt_docs.append({"text": f"创业项目案例：{p['project_name']}位于{f['city']}，所属赛道为{p['industry']}，当前处于{p['stage']}阶段，团队{p['team_size']}人，拥有{p['patent_count']}项知识产权，过去12个月客户数为{p['customer_count']}，营业收入为{p['revenue_12m']}元。创业孵化判断应结合团队、技术、市场验证、经营状态、融资需求及风险信息综合分析，不能仅凭单一指标下结论。"})
for pol in policies:
    cpt_docs.append({"text": f"创业政策示例：{pol['policy_name']}适用地区为{pol['region']}，面向产业{pol['industries']}，适用阶段{pol['target_stages']}，支持方式为{pol['support_type']}，最高支持金额{pol['amount_max']}元，有效期至{pol['valid_to']}。政策匹配需要先做资格硬条件校验，再做语义相关性排序。"})
write_jsonl(PROC / "startup_cpt_demo.jsonl", cpt_docs)

# -------------------- SFT generation --------------------
SYSTEM = (
    "你是创业精准孵化专家助手。依据给定事实做结构化分析；区分事实、规则和建议；"
    "不得编造未提供的数据；涉及信用、融资等高影响事项时只做辅助分析，不用敏感或无关个人属性作自动决策。"
)

def founder_input(idx: int):
    f = founders[idx % len(founders)]
    p = project_by_f[f["founder_id"]]
    prof = founder_profile[idx % len(founder_profile)]
    return f, p, prof

def j(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)

samples_by_task = defaultdict(list)

for i in range(TRAIN_PER_TASK + EVAL_PER_TASK):
    f, p, prof = founder_input(i)
    ve = venture_eval[i % len(venture_eval)]
    sk = skill_training[i % len(skill_training)]
    pm = policy_match[i % len(policy_match)]
    im = investment_match[i % len(investment_match)]
    ro = risk_opp[i % len(risk_opp)]
    ms = market_by_ind[p["industry"]][-1]

    # 1 founder profile
    inp = {"创业者": {k: f[k] for k in ["education","major","group_type","registration_status","venture_type","years_experience","city"]},
           "项目": {k: p[k] for k in ["industry","team_size","stage","patent_count","customer_count","revenue_12m"]}}
    out = {"当前阶段": p["stage"], "能力画像": {"技术": prof["technical_capability"], "商业": prof["commercial_capability"], "管理": prof["management_capability"]},
           "优势": ["行业/专业背景与项目方向具有一定相关性" if f["major"] in ["计算机","机械工程","电子信息","材料科学","生物工程"] else "具有跨领域创业背景", f"已有{p['team_size']}人团队"],
           "优先补足": [sk["weak_skill"]], "证据": [f"项目处于{p['stage']}", f"客户数{p['customer_count']}，近12月收入{p['revenue_12m']}元"]}
    samples_by_task["创业人员画像"].append(("请生成创业人员画像，并给出可验证证据。", j(inp), j(out)))

    # 2 capability improvement
    inp2 = {"阶段": p["stage"], "能力": {"技术": prof["technical_capability"], "商业": prof["commercial_capability"], "管理": prof["management_capability"]}, "目标": "未来90天提升项目推进效率"}
    out2 = {"首要能力缺口": sk["weak_skill"], "推荐课程": sk["course_name"], "90天行动": ["第1-30天完成课程与基线测评", "第31-60天在真实项目中完成一次应用", "第61-90天复测并根据结果调整"], "预期提升": f"约{sk['expected_improvement']}分（示例标签）"}
    samples_by_task["创业能力提升"].append(("基于创业阶段和能力缺口，生成能力提升建议。", j(inp2), j(out2)))

    # 3 opportunity identification
    inp3 = {"创业者专业": f["major"], "项目赛道": p["industry"], "市场信号": ms}
    opp = ro["opportunity_level"]
    out3 = {"机会等级": opp, "机会分数": ro["opportunity_score"], "主要信号": [f"融资增速{ms['funding_growth']}", f"专利增速{ms['patent_growth']}", f"岗位增速{ms['job_growth']}", f"政策强度{ms['policy_strength']}"], "验证动作": ["访谈10-20个潜在客户", "验证付费意愿", "用MVP验证核心需求"], "说明": "机会分数只用于筛选，最终需结合真实客户和竞争数据。"}
    samples_by_task["创业机会识别"].append(("识别当前创业机会，并给出验证动作。", j(inp3), j(out3)))

    # 4 path planning
    inp4 = {"阶段": p["stage"], "团队人数": p["team_size"], "客户数": p["customer_count"], "收入": p["revenue_12m"], "主要短板": sk["weak_skill"]}
    out4 = {"未来90天": ["明确核心客户与场景", "完成可量化的客户验证", f"补齐{sk['weak_skill']}能力"], "未来180天": ["形成可复制销售/交付流程", "建立关键经营指标看板"], "里程碑": [{"时间":"30天","指标":"完成10个客户访谈"},{"时间":"90天","指标":"形成至少1个可验证商业闭环"},{"时间":"180天","指标":"形成阶段复盘并决定扩张或调整"}]}
    samples_by_task["创业路径规划"].append(("生成分阶段、可执行的创业路径规划。", j(inp4), j(out4)))

    # 5 business plan generation
    inp5 = {"项目名称": p["project_name"], "赛道": p["industry"], "阶段": p["stage"], "团队": p["team_size"], "客户数": p["customer_count"], "收入": p["revenue_12m"], "专利": p["patent_count"]}
    out5 = {"执行摘要": f"{p['project_name']}聚焦{p['industry']}，当前处于{p['stage']}阶段。", "商业计划书结构": ["客户问题", "解决方案", "市场与竞争", "商业模式", "产品与技术", "团队", "里程碑", "财务与融资", "风险与对策"], "待补数据": ["目标客户画像", "客单价/毛利", "竞争对手", "获客成本", "现金流预测"], "提示": "缺失数据保留为待补项，不编造市场规模或财务数字。"}
    samples_by_task["创业计划书生成"].append(("根据现有资料生成商业计划书骨架，并明确缺失数据。", j(inp5), j(out5)))

    # 6 venture evaluation
    inp6 = {"项目": inp5, "风险": {"经营异常": ro["operating_abnormality"], "司法案件": ro["legal_cases"], "失信": ro["dishonesty_record"]}}
    out6 = {"综合分": ve["overall_score"], "分项": {"团队": ve["team_score"], "技术": ve["technology_score"], "市场": ve["market_score"], "验证": ve["traction_score"]}, "主要风险": ro["risk_level"], "证据": [f"客户数{p['customer_count']}", f"专利{p['patent_count']}项", f"近12月收入{p['revenue_12m']}元"], "使用限制": "分数为辅助评估，不应单独用于贷款、投资或就业等高影响决定。"}
    samples_by_task["创业项目评估"].append(("对创业项目做多维度评估，并说明证据和使用限制。", j(inp6), j(out6)))

    # 7 resource matching
    inp7 = {"项目": {"城市": f["city"], "行业": p["industry"], "阶段": p["stage"]}, "政策": pm}
    out7 = {"政策匹配": "匹配" if pm["eligible"] else "不完全匹配", "匹配分": pm["match_score"], "硬条件检查": {"地区": f["city"] == pm["policy_region"], "行业": p["industry"] in pm["policy_industries"].split("|"), "阶段": p["stage"] in pm["policy_target_stages"].split("|")}, "下一步": ["核对政策原文和有效期", "准备申报材料", "确认是否存在额外资格条件"]}
    samples_by_task["创业资源匹配"].append(("判断项目与政策/资源是否匹配，并先做硬条件校验。", j(inp7), j(out7)))

    # 8 risk assessment
    inp8 = {"项目": {"阶段": p["stage"], "收入": p["revenue_12m"], "客户数": p["customer_count"]}, "风险事件": {"经营异常": ro["operating_abnormality"], "司法案件": ro["legal_cases"], "失信": ro["dishonesty_record"]}, "市场增长": ro["market_growth"]}
    risk_items = []
    if ro["operating_abnormality"]: risk_items.append("经营异常")
    if ro["legal_cases"]: risk_items.append("司法风险")
    if ro["dishonesty_record"]: risk_items.append("失信风险")
    if not risk_items: risk_items.append("未发现高强度登记类风险信号")
    out8 = {"风险等级": ro["risk_level"], "风险分": ro["risk_score"], "风险项": risk_items, "建议": ["核验风险事件来源和时间", "建立现金流与合规月度监控", "重大合同和融资前做专项法律/财务尽调"], "说明": "这是基于给定字段的风险筛查，不替代专业法律、财务或征信结论。"}
    samples_by_task["创业风险评估"].append(("生成创业项目风险评估和处置建议。", j(inp8), j(out8)))

# Split each task independently to keep task balance.
train_sft = []
eval_sft = []
train_manifest = []
eval_manifest = []
for task, rows in samples_by_task.items():
    assert len(rows) == TRAIN_PER_TASK + EVAL_PER_TASK
    for n, (instr, inp, out) in enumerate(rows):
        rec = {"instruction": instr, "input": inp, "output": out, "system": SYSTEM}
        if n < TRAIN_PER_TASK:
            train_manifest.append({"row": len(train_sft), "task_type": task})
            train_sft.append(rec)
        else:
            eval_manifest.append({"row": len(eval_sft), "task_type": task})
            eval_sft.append(rec)

write_json(PROC / "startup_sft_train.json", train_sft)
write_json(PROC / "startup_sft_eval.json", eval_sft)
write_csv(PROC / "startup_sft_train_manifest.csv", train_manifest)
write_csv(PROC / "startup_sft_eval_manifest.csv", eval_manifest)

summary = {
    "seed": SEED,
    "raw": {
        "A_founders": len(founders), "B_employment_credit": len(employment), "C_projects": len(projects),
        "D_training_records": len(training_records), "E_policies": len(policies), "F_finance_risk": len(finance_risk),
        "F_investors": len(investors), "F_market_signals": len(market_signals)
    },
    "derived": {
        "founder_profile": len(founder_profile), "venture_stage_evaluation": len(venture_eval),
        "skill_gap_training": len(skill_training), "policy_resource_matching": len(policy_match),
        "investment_matching": len(investment_match), "risk_opportunity": len(risk_opp), "startup_cpt_demo": len(cpt_docs)
    },
    "sft": {"train": len(train_sft), "eval": len(eval_sft), "tasks": {k: len(v) for k, v in samples_by_task.items()}}
}
write_json(PROC / "generation_summary.json", summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
