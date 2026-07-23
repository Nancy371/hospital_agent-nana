"""Normalize inquiry and examination payloads into structured clinical evidence."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_NEGATION_RE = re.compile(
    r"(?:未见|未发现|未提示|不支持|无明显|无(?!力)|否认|排除|阴性|正常|完整|未检出|未培养出)"
)
_UNCERTAINTY_RE = re.compile(r"(?:考虑|可能|疑似|不能除外|倾向|待排)" )
_POSITIVE_RE = re.compile(r"(?:阳性|提示|符合|诊断为|检出|发现|可见|存在|增高|升高|降低|减低)" )
_SEVERITY_TERMS = ("极重度", "重度", "中重度", "中度", "轻中度", "轻度", "显著")
_ANATOMY_TERMS = (
    "左心房", "右心房", "左心室", "右心室", "二尖瓣", "三尖瓣", "肺动脉瓣",
    "主动脉瓣", "左肺", "右肺", "上肺", "下肺", "肾脏", "膀胱", "尿道",
    "肝脏", "胆囊", "胰腺", "脑", "视网膜", "脊柱", "关节",
)

_INTERPRETER_RULES: Tuple[Dict[str, Any], ...] = (
    {
        "finding": "dyspnea_on_exertion",
        "terms": (
            "\u8d70\u4e24\u6b65\u5c31\u5598",
            "\u8d70\u51e0\u6b65\u5c31\u5598",
            "\u6d3b\u52a8\u540e\u6c14\u77ed",
            "\u52b3\u529b\u6027\u547c\u5438\u56f0\u96be",
            "\u8fd0\u52a8\u540e\u6c14\u4fc3",
        ),
        "confidence": 0.9,
    },
    {
        "finding": "exercise_intolerance",
        "terms": (
            "\u8d70\u4e24\u6b65\u5c31\u5598",
            "\u6d3b\u52a8\u8010\u91cf\u4e0b\u964d",
            "\u8fd0\u52a8\u8010\u91cf\u4e0b\u964d",
            "\u52b3\u7d2f\u540e\u660e\u663e\u4e0d\u9002",
        ),
        "confidence": 0.88,
    },
    {
        "finding": "postprandial_nausea",
        "terms": (
            "\u996d\u540e\u6076\u5fc3",
            "\u8fdb\u98df\u540e\u6076\u5fc3",
            "\u9910\u540e\u6076\u5fc3",
        ),
        "confidence": 0.88,
    },
    {
        "finding": "periorbital_edema",
        "terms": (
            "\u65e9\u6668\u8d77\u5e8a\u773c\u775b\u80bf",
            "\u6668\u8d77\u773c\u7751\u6d6e\u80bf",
            "\u773c\u5468\u6d6e\u80bf",
            "\u773c\u7751\u6d6e\u80bf",
        ),
        "confidence": 0.9,
    },
    {
        "finding": "fluid_retention_pattern",
        "terms": (
            "\u65e9\u6668\u8d77\u5e8a\u773c\u775b\u80bf",
            "\u6668\u8d77\u773c\u7751\u6d6e\u80bf",
            "\u773c\u5468\u6d6e\u80bf",
            "\u4e0b\u80a2\u6c34\u80bf",
        ),
        "confidence": 0.84,
    },
    {
        "finding": "polydipsia",
        "terms": (
            "\u559d\u5f88\u591a\u6c34",
            "\u996e\u6c34\u660e\u663e\u589e\u591a",
            "\u591a\u996e",
            "\u53e3\u6e34\u591a\u996e",
        ),
        "confidence": 0.9,
    },
    {
        "finding": "tropical_exposure",
        "terms": (
            "\u70ed\u5e26\u66b4\u9732",
            "\u70ed\u5e26\u5730\u533a",
            "\u5357\u65b9\u519c\u6751",
            "\u519c\u6751\u513f\u7ae5",
            "\u519c\u6751\u73af\u5883",
        ),
        "confidence": 0.88,
    },
    {
        "finding": "deep_skin_ulcer",
        "terms": (
            "\u6df1\u90e8\u6e83\u75a1",
            "\u76ae\u80a4\u6e83\u75a1",
            "\u6e83\u75a1\u6027\u76ae\u635f",
            "\u6e17\u51fa\u6027\u6e83\u75a1",
        ),
        "confidence": 0.9,
    },
    {
        "finding": "crusted_skin_lesion",
        "terms": (
            "\u7ed3\u75c2",
            "\u75c2\u76ae",
            "\u9ec4\u8272\u75c2\u76ae",
            "\u6e17\u51fa\u540e\u7ed3\u75c2",
        ),
        "confidence": 0.86,
    },
    {
        "finding": "periostitis",
        "terms": (
            "\u9aa8\u819c\u708e",
            "\u9aa8\u819c\u589e\u539a",
            "\u9aa8\u75db",
            "\u591c\u95f4\u9aa8\u75db",
        ),
        "confidence": 0.9,
    },
    {
        "finding": "treponemal_disease_pattern",
        "terms": (
            "\u96c5\u53f8\u75c5",
            "\u8815\u65cb\u4f53",
            "\u6885\u6bd2\u8840\u6e05\u5b66\u9633\u6027",
            "\u975e\u6027\u75c5\u6027\u6885\u6bd2\u87ba\u65cb\u4f53",
        ),
        "confidence": 0.92,
    },
)


# These findings are reusable clinical concepts rather than disease-specific rules.
_PHRASE_FINDINGS: Dict[str, Tuple[str, ...]] = {
    "cough": ("咳嗽", "干咳", "咳痰"),
    "fever": ("发热", "高热", "低热", "发烧"),
    "night_sweats": ("盗汗", "夜间盗汗", "晚上出汗", "夜间出汗"),
    "acute_course": ("1天", "2天", "3天", "急性起病", "突然开始", "受凉后"),
    "rhinorrhea": ("流鼻涕", "流涕", "鼻涕变稠", "清涕"),
    "nasal_congestion": ("鼻塞", "张口呼吸", "鼻子不通气"),
    "wheeze": ("喘息", "哮鸣音", "喘"),
    "dyspnea": ("呼吸困难", "气短", "气促", "喘不上气", "呼吸急促"),
    "hypoxemia": ("低氧血症", "血氧下降", "氧饱和度下降", "SpO2降低", "SpO₂降低"),
    "hemoptysis": ("咯血", "血痰", "咳血"),
    "arthralgia": ("关节痛", "关节疼痛", "关节肿痛"),
    "joint_stiffness": ("关节僵硬", "晨僵", "膝关节僵硬"),
    "trauma_history": ("外伤史", "创伤史", "受伤后", "扭伤后", "跌倒后", "摔伤后", "撞伤后"),
    "post_traumatic_joint_pain": ("创伤后关节痛", "外伤后关节痛", "受伤后关节痛"),
    "activity_related_joint_pain": ("活动后关节痛", "活动后疼痛", "负重后疼痛", "上下楼疼痛"),
    "mechanical_overload_trigger": ("长时间步行", "步行距离增加", "爬楼梯", "负重", "运动后加重", "机械负荷"),
    "joint_space_narrowing": ("关节间隙变窄", "关节间隙狭窄"),
    "osteophyte": ("骨赘", "骨刺", "边缘骨质增生"),
    "weakness": ("乏力", "无力", "全身无力"),
    "dizziness": ("头晕", "眩晕"),
    "syncope": ("晕厥", "昏厥", "晕倒"),
    "presyncope": ("近晕厥", "晕厥前兆", "快要晕倒", "眼前发黑"),
    "bradycardia": ("心动过缓", "心率慢", "脉搏慢", "心率过慢"),
    "av_block": ("房室传导阻滞", "传导阻滞"),
    "second_degree_av_block": (
        "二度房室传导阻滞",
        "Ⅱ度房室传导阻滞",
        "II度房室传导阻滞",
        "二度AV传导阻滞",
        "Mobitz",
        "Wenckebach",
    ),
    "pr_prolongation": ("PR间期延长", "PR 间期延长", "PR延长"),
    "dropped_beats": ("漏搏", "脱落搏动", "P波未下传", "QRS波群脱落"),
    "palpitation": ("心悸", "心慌"),
    "muscle_cramp": ("抽筋", "肌肉痉挛", "手足搐搦"),
    "dark_urine": ("尿色变深", "深色尿", "酱油色尿"),
    "leg_edema": ("下肢水肿", "腿肿", "双下肢水肿"),
    "orthopnea": ("端坐呼吸", "不能平卧", "高枕卧位"),
    "paroxysmal_nocturnal_dyspnea": ("夜间阵发性呼吸困难",),
    "choking_event": ("呛咳", "误吸", "进食后呛", "喝水呛"),
    "dysphagia": ("吞咽困难", "喂养困难"),
    "cyanosis": ("发绀", "口唇发绀", "口周发绀", "青紫"),
    "feeding_diaphoresis": ("吃奶出汗", "喂奶出汗", "喂养时出汗", "喂奶时多汗"),
    "aspiration_risk": ("慢性误吸", "反复呛奶", "吞咽功能障碍"),
    "microscopic_hematuria": ("镜下血尿", "显微镜下血尿", "尿红细胞增多", "红细胞管型"),
    "proteinuria": ("蛋白尿", "尿蛋白阳性"),
    "pulmonary_hemorrhage": ("肺泡出血", "肺出血", "弥漫性肺泡出血"),
    "anca_positive": (
        "ANCA阳性",
        "ANCA 阳性",
        "ANCA谱阳性",
        "抗中性粒细胞胞浆抗体阳性",
        "抗中性粒细胞胞质抗体阳性",
    ),
    "mpo_anca_positive": (
        "MPO-ANCA阳性",
        "MPO-ANCA 阳性",
        "MPO阳性",
        "MPO抗体阳性",
        "抗MPO阳性",
        "抗髓过氧化物酶抗体阳性",
        "髓过氧化物酶抗体阳性",
    ),
    "p_anca_positive": ("p-ANCA阳性", "P-ANCA阳性", "p-ANCA 阳性"),
    "renal_impairment": ("肾功能受损", "肌酐升高", "肾小球滤过率降低"),
    "jaundice": ("黄疸", "皮肤黄染"),
    "scleral_icterus": ("巩膜黄染", "眼白发黄"),
    "neonatal_jaundice": ("新生儿黄疸", "出生后黄疸", "婴儿黄疸"),
    "bilirubin_high": ("胆红素升高", "高胆红素血症"),
    "unconjugated_hyperbilirubinemia": ("间接胆红素升高", "非结合胆红素升高", "未结合胆红素升高"),
    "ugt1a1_positive": (
        "UGT1A1基因突变",
        "UGT1A1突变",
        "UGT1A1双等位致病变异",
        "UGT1A1 双等位致病变异",
        "UGT1A1致病变异",
        "UGT1A1变异",
    ),
    "genetic_suspicion": ("遗传", "家族史", "基因异常", "基因突变"),
    "poor_feeding": ("喂养差", "吃奶差", "拒奶"),
    "lethargy": ("嗜睡", "反应差", "精神差"),
    "low_magnesium": ("低镁血症", "血镁降低", "血镁偏低", "镁降低"),
    "low_urine_magnesium": ("尿镁降低", "尿镁偏低", "24小时尿镁降低"),
    "magnesium_load_retention_high": ("镁负荷保留率升高", "镁保留率升高"),
    "magnesium_depletion": ("镁缺乏", "镁储备不足"),
    "vitamin_d_low": ("维生素D缺乏", "25羟维生素D降低", "25-OH-D降低"),
    "alp_elevated": ("碱性磷酸酶升高", "ALP升高", "ALP 增高"),
    "hypocalcemia": ("低钙血症", "血钙降低", "血钙偏低"),
    "bone_deformity": ("骨骼畸形", "O型腿", "X型腿", "肋骨串珠", "方颅"),
    "waddling_gait": ("鸭步", "摇摆步态", "步态异常"),
    "mitral_regurgitation": ("二尖瓣反流", "二尖瓣返流"),
    "tricuspid_regurgitation": ("三尖瓣反流", "三尖瓣返流"),
    "cardiac_murmur": ("心脏杂音", "收缩期杂音", "胸骨左缘杂音", "全收缩期杂音"),
    "dextrocardia": ("右位心", "心脏右位", "右位心影"),
    "right_apex_beat": ("心尖搏动右移", "右侧心尖搏动", "心尖搏动位于右侧"),
    "mirror_image_ecg": ("胸前导联R波进展反向", "镜像心电图", "右位心心电图"),
    "pulmonary_valve_stenosis": ("肺动脉瓣狭窄",),
    "congenital_heart_defect": ("先天性心脏病", "先心病", "先天性心脏缺陷", "先天性缺损"),
    "ventricular_septal_defect": ("室间隔缺损", "大型室间隔缺损", "VSD"),
    "right_to_left_shunt": ("右向左分流", "右至左分流"),
    "pulmonary_hypertension": ("肺动脉高压", "肺动脉压升高"),
    "right_ventricular_hypertrophy": ("右心室肥厚", "右室肥厚"),
    "atrial_septal_defect": ("房间隔缺损", "继发孔型房缺", "ASD"),
    "left_to_right_shunt": ("左向右分流", "左至右分流"),
    "right_heart_enlargement": ("右心房增大", "右心室扩大", "右心室扩张", "右心扩大"),
    "heart_failure_state": ("心力衰竭", "心衰", "容量超负荷"),
    "atelectasis": ("肺不张", "肺叶不张", "肺段不张"),
    "bronchopneumonia": (
        "支气管肺炎",
        "小叶性肺炎",
        "支气管肺炎样",
        "支气管周围斑片状",
        "脓性分泌物",
    ),
    "pneumonia_infiltrate": (
        "肺部浸润",
        "肺实变",
        "实变",
        "斑片状阴影",
        "肺炎影像",
        "空气支气管征",
        "支气管充气征",
    ),
    "rib_fracture": ("肋骨骨折", "肋骨断裂"),
    "oliguria": ("少尿", "尿量减少"),
    "eyelid_edema": ("眼睑水肿", "眼皮水肿"),
    "pruritus": ("皮肤瘙痒", "瘙痒"),
    "vesicular_rash": ("水疱", "小水疱", "小泡泡", "疱疹样皮疹", "水痘样皮疹", "成批水疱"),
    "childcare_exposure": ("幼儿园接触", "同班小朋友", "学校接触", "水痘接触史", "接触水痘", "接触确诊患儿"),
    "maculopapular_rash": ("丘疹", "斑丘疹", "红色丘疹", "皮肤丘疹"),
    "pericarditic_chest_pain": ("深呼吸加重", "平卧加重", "前倾缓解", "坐起前倾缓解", "胸膜性胸痛"),
    "pericardial_effusion": ("心包积液", "心包腔积液"),
    "pericardial_thickening": ("心包增厚", "心包膜增厚"),
    "tuberculosis_exposure": ("结核接触", "结核病接触", "接触确诊患者", "接触肺结核患者"),
    "post_icu_state": ("刚从ICU出来", "近期ICU", "ICU出院", "重症监护后"),
    "immunocompromised": ("免疫抑制", "免疫低下", "长期激素", "化疗后", "移植后", "中性粒细胞减少"),
    "candida_positive": ("念珠菌", "白色念珠菌", "假丝酵母菌", "Candida"),
    "fungal_pneumonia": ("真菌性肺炎", "肺真菌感染", "真菌病原体"),
    "uremia": ("尿毒症", "终末期肾病", "ESRD"),
    "egfr_low": ("eGFR降低", "eGFR下降", "肾小球滤过率降低"),
    "urea_elevated": ("尿素氮升高", "BUN升高"),
    "hyperkalemia": ("高钾血症", "血钾升高", "钾升高"),
    "metabolic_acidosis": ("代谢性酸中毒", "碳酸氢根降低"),
    "ascites": ("腹水",),
    "abdominal_distension": ("腹胀",),
    "ovarian_enlargement": ("卵巢增大", "卵巢体积增大", "多囊样卵巢", "卵巢过度刺激"),
    "hemoconcentration": ("血液浓缩", "红细胞压积升高", "血细胞比容升高", "HCT升高"),
    "hypoalbuminemia": ("低白蛋白", "白蛋白降低", "白蛋白偏低"),
    "ohss_risk": ("促排卵", "取卵", "试管婴儿", "辅助生殖", "hCG"),
    "portal_vein_dilation": ("门静脉增宽", "门静脉内径增宽", "门静脉扩张"),
    "portal_flow_abnormal": ("门静脉血流", "门脉血流", "门静脉血流速度降低"),
    "varices": ("食管胃底静脉曲张", "静脉曲张"),
    "thrombocytopenia": ("血小板减少", "血小板降低"),
    "detrusor_overactivity": ("逼尿肌过度活动",),
    "urinary_incontinence": ("尿失禁", "漏尿"),
    "stress_urinary_incontinence": ("压力性尿失禁", "压力性漏尿", "腹压性尿失禁"),
    "urine_leak_with_pressure": ("咳嗽漏尿", "喷嚏漏尿", "大笑漏尿", "运动漏尿", "腹压增加漏尿", "腹压增加时漏尿"),
    "urinary_urgency": ("尿急",),
    "urinary_frequency": ("尿频",),
    "dysuria": ("尿痛", "排尿烧灼", "排尿时烧灼", "烧灼感"),
    "perineal_pain": ("会阴痛", "会阴部疼痛", "会阴胀痛"),
    "pelvic_pain": ("盆腔痛", "下腹痛", "耻骨上疼痛"),
    "prostate_tenderness": ("前列腺压痛", "直肠指检压痛", "DRE压痛"),
    "pyuria": ("脓尿", "尿白细胞增多", "白细胞尿"),
    "bacteriuria": ("菌尿", "尿细菌增多"),
    "urine_culture_positive": ("尿培养阳性", "尿培养检出"),
    "urine_culture_no_growth": ("尿培养阴性", "尿培养无生长", "未培养出细菌"),
    "neutropenia": ("中性粒细胞减少", "粒细胞减少"),
    "splenomegaly": ("脾大", "脾脏增大"),
    "cor_triatriatum": ("三房心", "左心房隔膜", "左房隔膜", "限制性开窗"),
    "atrioventricular_septal_defect": ("心内膜垫缺损", "房室间隔缺损", "完全性房室间隔缺损", "共同房室瓣"),
    "pulmonary_venous_obstruction": ("肺静脉回流受阻", "左房流入受阻", "肺静脉梗阻"),
    "lacrimal_gland_swelling": ("泪腺肿大", "泪腺区肿胀", "眼睑外上方肿胀", "外上眶肿胀"),
    "lacrimal_gland_pain": ("泪腺区疼痛", "眼眶外上方疼痛", "眼部压迫感"),
    "tearing": ("流泪", "溢泪"),
    "heartburn": ("烧心", "反酸", "胃酸反流"),
    "retrosternal_burning": ("胸骨后烧灼", "胸口烧灼", "胸骨后烧灼样疼痛"),
    "odynophagia": ("吞咽痛", "吞咽疼痛"),
    "esophageal_ulcer": ("食管溃疡", "食道溃疡", "食管黏膜溃疡", "食管黏膜糜烂溃疡"),
    "vaginal_bleeding": ("阴道出血", "阴道流血", "少量阴道流血"),
    "early_pregnancy": ("早孕", "妊娠早期", "停经", "怀孕", "孕早期"),
    "hcg_positive": ("β-hCG阳性", "血清β-hCG升高", "hCG阳性", "妊娠试验阳性"),
    "progesterone_low": ("孕酮降低", "孕酮偏低"),
    "oligomenorrhea": ("月经稀发", "月经不规律", "闭经", "月经紊乱"),
    "hyperandrogenism": ("高雄激素", "多毛", "痤疮", "雄激素升高"),
    "polycystic_ovaries": ("多囊卵巢", "多囊样卵巢", "卵巢多囊样改变"),
    "treponemal_skin_lesion": ("莓疮", "雅司", "乳头瘤样皮损", "树莓样皮损", "湿疣样丘疹"),
    "treponema_positive": ("梅毒螺旋体颗粒凝集试验阳性", "TPPA阳性", "螺旋体阳性", "Treponema"),
    "iris_coloboma": ("虹膜缺损", "虹膜裂隙", "钥匙孔样瞳孔", "虹膜缺损畸形"),
    "photophobia": ("畏光", "怕光"),
    "night_vision_decline": ("夜视力下降", "夜间视力差"),
    "visual_blurring": ("视物模糊", "看东西模糊", "视力下降", "看不清"),
    "lens_dislocation": ("晶状体脱位", "晶状体半脱位", "晶状体位置异常"),
    "dermatomal_pain": ("沿神经分布疼痛", "带状疼痛", "烧灼样皮痛"),
    "renal_colic": ("肾绞痛", "腰腹部绞痛", "阵发性绞痛"),
    "renal_stone": ("肾结石", "输尿管结石", "泌尿系结石"),
    "triple_x_karyotype": ("47,XXX", "X三体", "Triple X", "超雌综合征"),
    "tall_stature": ("身材高大", "高身材", "身高偏高"),
    "premature_ovarian_insufficiency": ("卵巢功能不全", "卵巢早衰", "卵巢储备下降"),
    "adrenal_insufficiency": ("肾上腺功能不全", "Addison", "阿狄森", "肾上腺皮质功能减退"),
    "orthostatic_hypotension": ("体位性低血压", "直立性低血压", "站起时头晕", "卧立位血压下降"),
    "hyperpigmentation": ("皮肤色素沉着", "色素沉着", "皮肤变黑"),
    "cortisol_low": ("皮质醇降低", "血清皮质醇低", "8AM皮质醇低"),
    "acth_high": ("ACTH升高", "促肾上腺皮质激素升高"),
    "hyponatremia": ("低钠血症", "血钠降低", "钠降低"),
    "near_vision_difficulty": ("看近模糊", "近距离看不清", "阅读困难", "填表困难", "看手机费劲"),
    "age_related_near_blur": ("老花", "老视", "年龄相关调节不足", "中老年看近模糊"),
    "refractive_error": ("屈光不正", "屈光异常", "验光异常", "需要阅读眼镜", "+1.50D"),
    "ambiguous_genitalia": ("外生殖器发育异常", "性别发育异常", "生殖器不典型", "阴蒂肥大", "尿道下裂"),
    "sex_development_disorder": ("性发育异常", "性别发育异常", "DSD", "性腺发育异常"),
    "ovotesticular_tissue": ("卵睾组织", "卵巢和睾丸组织", "卵睾性"),
    "karyotype_mosaic": ("嵌合核型", "染色体嵌合", "46,XX/46,XY", "46XX/46XY"),
    "cryptorchidism": ("隐睾", "睾丸未降"),
    "hypospadias": ("尿道下裂",),
    "jaw_locked_open": ("张口后不能闭口", "嘴巴合不上", "下巴合不上", "张口不能闭合"),
    "unable_close_mouth": ("不能闭口", "闭不上嘴", "口不能闭合"),
    "preauricular_pain": ("耳前区疼痛", "颞下颌关节疼痛", "下颌关节疼痛"),
    "tmj_dislocation": ("颞下颌关节脱位", "下颌关节脱位", "TMJ脱位"),
    "malocclusion": ("咬合错乱", "咬合不齐"),
    "anogenital_warts": ("肛周疣体", "外阴疣体", "生殖器疣", "肛生殖器疣", "尖锐湿疣"),
    "cauliflower_lesions": ("菜花样赘生物", "菜花样疣体", "乳头状赘生物", "疣状赘生物"),
    "hpv_related_lesions": ("HPV阳性", "HPV疣", "人乳头瘤病毒"),
    "frothy_vaginal_discharge": ("泡沫样分泌物", "黄绿色泡沫样白带", "泡沫样白带"),
    "vaginal_pruritus": ("外阴瘙痒", "阴道瘙痒"),
    "strawberry_cervix": ("草莓样宫颈", "宫颈点状出血"),
    "trichomonas_positive": ("滴虫阳性", "阴道毛滴虫", "毛滴虫阳性", "Trichomonas"),
    "vaginal_ph_high": ("阴道pH升高", "阴道pH>4.5", "pH大于4.5"),
    "umbilical_discharge": ("脐部流液", "脐部渗液", "脐部流脓", "脐孔流液"),
    "umbilical_mass": ("脐部肿块", "脐下肿块", "脐周包块", "下腹正中包块", "下腹正中肿块"),
    "midline_suprapubic_cyst": ("下腹正中囊性肿物", "膀胱顶部囊性肿物", "脐尿管残余"),
    "urachal_cyst_imaging": ("脐尿管囊肿", "脐尿管残余囊肿", "urachal cyst"),
    "rural_child_contact": ("农村接触", "乡村接触", "儿童密切接触", "共用毛巾", "其他儿童类似皮损"),
    "crusted_exudative_skin_ulcer": ("结痂流黄水", "渗出结痂", "乳头瘤样皮损", "树莓样皮损", "慢性结痂皮损"),
    "regional_lymphadenopathy": ("腹股沟淋巴结肿大", "局部淋巴结肿大", "区域淋巴结肿大"),
    "treponemal_serology_positive": ("梅毒血清学阳性", "螺旋体血清学阳性", "TPPA阳性", "RPR阳性"),
    "leukocoria": ("白瞳", "瞳孔发白", "猫眼反光"),
    "intraocular_mass": ("眼内肿物", "眼内占位", "视网膜肿瘤"),
    "nasopharyngeal_foreign_body_sensation": ("咽部异物感", "鼻咽异物感", "咽喉异物感"),
    "throat_dryness": ("咽干", "咽部干燥", "鼻咽干燥"),
    "throat_clearing": ("清嗓", "频繁清嗓", "反复清嗓"),
    "chronic_course": ("慢性", "反复", "长期", "迁延"),
    "nasopharyngoscopy_abnormal": ("鼻咽镜异常", "鼻咽黏膜充血", "鼻咽部充血", "鼻咽部淋巴滤泡"),
    "nasopharyngeal_chronic_inflammation": (
        "鼻咽慢性炎症",
        "鼻咽部慢性炎症",
        "鼻咽黏膜慢性炎症",
        "鼻咽炎症改变",
    ),
    "cytology_chronic_inflammation": (
        "脱落细胞学提示慢性炎症",
        "脱落细胞学慢性炎症",
        "细胞学慢性炎症",
    ),
    "microtia": ("小耳畸形", "小耳"),
    "auricular_malformation": ("耳廓畸形", "耳郭畸形", "耳廓发育不良"),
    "external_auditory_canal_atresia": ("外耳道闭锁", "外耳道狭窄"),
    "congenital_onset": ("出生即有", "出生后即发现", "先天", "出生时"),
    "hearing_loss": ("听力下降", "听力差", "听力异常"),
    "tinnitus": ("耳鸣",),
    "ear_pain": ("耳痛", "耳部疼痛", "耳朵痛"),
    "ear_fullness": ("耳闷", "耳堵", "耳胀"),
    "ear_canal_irritation_trigger": ("棉签掏耳", "掏耳", "挖耳", "耳道刺激", "外耳道刺激", "采耳后", "清理耳道后"),
    "acute_tympanitis": ("急性鼓膜炎", "鼓膜炎"),
    "tympanic_membrane_inflammation": ("鼓膜充血", "鼓膜红肿", "鼓膜炎症", "鼓膜明显充血", "鼓膜红斑"),
    "tympanic_bulla": ("鼓膜疱疹", "鼓膜大疱", "鼓膜水疱", "大疱性鼓膜炎"),
    "abr_abnormal": ("ABR异常", "听性脑干反应异常"),
    "temporal_bone_ct_abnormal": ("颞骨CT异常", "颞骨发育异常"),
}

_NEGATIVE_FACT_FINDINGS = {
    "urine_culture_no_growth",
    "leukocyte_esterase_negative",
    "nitrite_negative",
    "urine_wbc_normal",
    "normal_postvoid_residual",
}


@dataclass
class Observation:
    finding: str
    source: str
    value: Optional[float] = None
    unit: str = ""
    reference_range: str = ""
    direction: str = ""
    polarity: str = "positive"
    severity: str = ""
    anatomy: str = ""
    temporality: str = ""
    confidence: float = 0.8
    raw_text: str = ""
    field_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.value is None:
            data.pop("value", None)
        return data


@dataclass
class EvidenceBundle:
    observations: List[Observation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Any) -> "EvidenceBundle":
        if not isinstance(value, dict):
            return cls()
        fields = set(Observation.__dataclass_fields__)
        observations: List[Observation] = []
        for item in value.get("observations", []) or []:
            if not isinstance(item, dict) or not item.get("finding"):
                continue
            payload = {key: item[key] for key in fields if key in item}
            try:
                observations.append(Observation(**payload))
            except (TypeError, ValueError):
                continue
        return cls(observations=observations)

    def to_dict(self) -> Dict[str, Any]:
        return {"observations": [item.to_dict() for item in self.observations]}

    def to_graph(self) -> "EvidenceGraph":
        return EvidenceGraph.from_bundle(self)

    def positive(self) -> List[Observation]:
        return [item for item in self.observations if item.polarity == "positive"]

    def major(self) -> List[Observation]:
        specific_paths = {
            (item.source, item.field_path)
            for item in self.positive()
            if not item.finding.startswith(("field:", "symptom:"))
        }
        return [
            item
            for item in self.positive()
            if item.confidence >= 0.75
            and not item.finding.startswith("field:")
            and not (
                item.finding.startswith("symptom:")
                and (item.source, item.field_path) in specific_paths
            )
            and item.finding not in {"weakness", "dizziness", "anxiety"}
        ]

    def findings(self, polarity: Optional[str] = None) -> List[str]:
        items = self.observations
        if polarity:
            items = [item for item in items if item.polarity == polarity]
        return list(dict.fromkeys(item.finding for item in items))

    def render_summary(self, limit: int = 24) -> str:
        lines = ["【结构化临床证据】"]
        ranked = sorted(
            self.observations,
            key=lambda item: (item.polarity == "positive", item.confidence),
            reverse=True,
        )
        for item in ranked[:limit]:
            value = ""
            if item.value is not None:
                value = f" value={item.value}{item.unit}"
            direction = f" direction={item.direction}" if item.direction else ""
            lines.append(
                f"- {item.finding} source={item.source} polarity={item.polarity}"
                f" confidence={item.confidence:.2f}{value}{direction}"
            )
        return "\n".join(lines)

    def to_query(self, limit: int = 30) -> str:
        terms: List[str] = []
        for item in sorted(self.positive(), key=lambda x: x.confidence, reverse=True):
            terms.extend((item.finding, item.source))
            if item.value is not None:
                terms.append(f"{item.finding}={item.value:g}{item.unit}")
            if item.direction:
                terms.append(item.direction)
            if item.severity:
                terms.append(item.severity)
            if item.anatomy:
                terms.append(item.anatomy)
            if (item.direction or item.severity or item.value is not None) and item.raw_text:
                terms.append(item.raw_text[:120])
            if len(terms) >= limit:
                break
        return " ".join(dict.fromkeys(term for term in terms if term))


@dataclass
class EvidenceGraph:
    symptoms: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    physical: List[Dict[str, Any]] = field(default_factory=list)
    labs: List[Dict[str, Any]] = field(default_factory=list)
    imaging: List[Dict[str, Any]] = field(default_factory=list)
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    red_flags: List[Dict[str, Any]] = field(default_factory=list)
    negative_findings: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    bundle: EvidenceBundle = field(default_factory=EvidenceBundle, repr=False)

    @classmethod
    def from_bundle(cls, bundle: EvidenceBundle) -> "EvidenceGraph":
        graph = cls(bundle=bundle, observations=[item.to_dict() for item in bundle.observations])
        for item in bundle.observations:
            row = item.to_dict()
            if item.polarity == "negative":
                graph.negative_findings.append(row)
                continue
            if item.finding.startswith("symptom:") or item.source == "问诊":
                graph.symptoms.append(row)
            if _is_history_path(item.field_path):
                graph.history.append(row)
            if _is_physical_source(item.source):
                graph.physical.append(row)
            if _is_lab_source(item.source):
                graph.labs.append(row)
            if _is_imaging_source(item.source):
                graph.imaging.append(row)
            if _is_risk_factor(item):
                graph.risk_factors.append(row)
            if item.confidence >= 0.85 and not item.finding.startswith("field:"):
                graph.red_flags.append(row)
        return graph

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symptoms": list(self.symptoms),
            "history": list(self.history),
            "physical": list(self.physical),
            "labs": list(self.labs),
            "imaging": list(self.imaging),
            "risk_factors": list(self.risk_factors),
            "red_flags": list(self.red_flags),
            "negative_findings": list(self.negative_findings),
            "observations": list(self.observations),
        }


class EvidenceAgent:
    """Build an Evidence Graph before diagnosis generation and ranking."""

    def __init__(self, ref_dir: str = "data/ref_data", normalizer: Optional[ClinicalEvidenceNormalizer] = None):
        self.normalizer = normalizer or ClinicalEvidenceNormalizer(ref_dir=ref_dir)

    def build_graph(
        self,
        collected_info: Optional[Dict[str, Any]],
        exam_results: Optional[Dict[str, Any]],
    ) -> EvidenceGraph:
        return self.normalizer.normalize(collected_info, exam_results).to_graph()


class ClinicalEvidenceInterpreter:
    """Translate patient language into reusable clinical findings."""

    def interpret(
        self,
        collected_info: Optional[Dict[str, Any]],
        exam_results: Optional[Dict[str, Any]],
    ) -> List[Observation]:
        text = self._case_text(collected_info, exam_results)
        if not text:
            return []
        observations: List[Observation] = []
        matched_findings: set = set()
        for rule in _INTERPRETER_RULES:
            finding = str(rule.get("finding") or "").strip()
            terms = tuple(str(item) for item in rule.get("terms") or ())
            term = next((item for item in terms if item and item in text), "")
            if not finding or not term:
                continue
            matched_findings.add(finding)
            observations.append(
                Observation(
                    finding=finding,
                    source="evidence_interpreter",
                    polarity="positive",
                    confidence=float(rule.get("confidence", 0.86) or 0.86),
                    raw_text=term,
                    field_path=f"interpreter.{finding}",
                )
            )

        # Composite interpretations are deliberately conservative: they describe
        # a clinical pattern, not a diagnosis, and still require the Judge to decide.
        if {"deep_skin_ulcer", "crusted_skin_lesion"} & matched_findings and (
            {"tropical_exposure", "periostitis"} & matched_findings
        ):
            observations.append(
                Observation(
                    finding="treponemal_skin_lesion",
                    source="evidence_interpreter",
                    polarity="positive",
                    confidence=0.9,
                    raw_text="skin ulcer pattern with compatible exposure or periostitis",
                    field_path="interpreter.treponemal_skin_lesion",
                )
            )
        if "dyspnea_on_exertion" in matched_findings:
            observations.append(
                Observation(
                    finding="cardiopulmonary_exertional_pattern",
                    source="evidence_interpreter",
                    polarity="positive",
                    confidence=0.82,
                    raw_text="dyspnea on exertion",
                    field_path="interpreter.cardiopulmonary_exertional_pattern",
                )
            )
        return observations

    @staticmethod
    def _case_text(
        collected_info: Optional[Dict[str, Any]],
        exam_results: Optional[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []
        for _path, value in _flatten_leaves(collected_info or {}):
            parts.append(_stringify(value))
        for exam_name, payload in (exam_results or {}).items():
            parts.append(str(exam_name))
            for _path, value in _flatten_leaves(payload):
                parts.append(_stringify(value))
        return " ".join(part for part in parts if part)


class ClinicalEvidenceNormalizer:
    """Convert heterogeneous nested payloads into reusable observations."""

    def __init__(self, ref_dir: str = "data/ref_data"):
        self.ref_dir = ref_dir
        self.diagnosis_aliases = self._load_diagnosis_aliases()
        self.interpreter = ClinicalEvidenceInterpreter()

    def normalize(
        self,
        collected_info: Optional[Dict[str, Any]],
        exam_results: Optional[Dict[str, Any]],
    ) -> EvidenceBundle:
        observations: List[Observation] = []
        info = collected_info or {}
        exams = exam_results or {}

        for index, symptom in enumerate(_as_text_list(info.get("symptoms"))):
            field_path = f"symptoms.{index}"
            observations.append(
                Observation(
                    finding=f"symptom:{_normalize_term(symptom)}",
                    source="问诊",
                    confidence=0.82,
                    raw_text=symptom,
                    field_path=field_path,
                )
            )
            observations.extend(self._phrase_observations("问诊", symptom, field_path))

        for path, value in _flatten_leaves(info):
            if path == "symptoms" or path.startswith("symptoms."):
                continue
            text = f"{path}: {_stringify(value)}"
            observations.extend(self._leaf_observations("问诊", path, text))

        for exam_name, payload in exams.items():
            if isinstance(payload, dict) and payload.get("status") == "invalid":
                continue
            for path, value in _flatten_leaves(payload):
                if path in {"status", "abnormal_indicators"}:
                    continue
                text = f"{path}: {_stringify(value)}"
                observations.extend(self._leaf_observations(str(exam_name), path, text))
            if isinstance(payload, dict):
                for indicator in _as_text_list(payload.get("abnormal_indicators")):
                    observations.extend(
                        self._leaf_observations(
                            str(exam_name),
                            "abnormal_indicators",
                            f"异常指标: {indicator}",
                            forced_positive=True,
                        )
                    )

        observations.extend(self.interpreter.interpret(info, exams))
        return EvidenceBundle(self._dedupe(observations))

    def _leaf_observations(
        self,
        source: str,
        path: str,
        text: str,
        forced_positive: bool = False,
    ) -> List[Observation]:
        polarity, confidence = self._polarity(text)
        if forced_positive:
            polarity, confidence = "positive", max(confidence, 0.85)
        value, unit, reference, direction = self._numeric_details(text)
        severity = next((term for term in _SEVERITY_TERMS if term in text), "")
        anatomy = _extract_anatomy(text)
        temporality = _extract_temporality(text)
        observations = [
            Observation(
                finding=f"field:{_normalize_term(path.split('.')[-1])}",
                source=source,
                value=value,
                unit=unit,
                reference_range=reference,
                direction=direction,
                polarity=polarity,
                severity=severity,
                anatomy=anatomy,
                temporality=temporality,
                confidence=confidence,
                raw_text=text,
                field_path=path,
            )
        ]
        observations.extend(self._phrase_observations(source, text, path))
        observations.extend(self._diagnosis_mentions(source, text, path))
        observations.extend(self._semantic_lab_observations(source, path, text))
        observations.extend(
            self._numeric_clinical_observations(
                source, path, text, value, unit, reference, direction, polarity
            )
        )
        return observations

    def _phrase_observations(self, source: str, text: str, path: str) -> List[Observation]:
        observations: List[Observation] = []
        for finding, terms in _PHRASE_FINDINGS.items():
            term = next((item for item in terms if item.lower() in text.lower()), None)
            if not term:
                continue
            if _is_reference_only_mention(text, term):
                continue
            polarity, confidence = self._polarity(text, term)
            if finding in _NEGATIVE_FACT_FINDINGS:
                polarity, confidence = "positive", max(confidence, 0.9)
            observations.append(
                Observation(
                    finding=finding,
                    source=source,
                    polarity=polarity,
                    severity=next((item for item in _SEVERITY_TERMS if item in text), ""),
                    anatomy=_extract_anatomy(text),
                    temporality=_extract_temporality(text),
                    confidence=confidence,
                    raw_text=text,
                    field_path=path,
                )
            )
        return observations

    def _diagnosis_mentions(self, source: str, text: str, path: str) -> List[Observation]:
        result: List[Observation] = []
        matched_aliases = [
            alias for alias in self.diagnosis_aliases
            if _contains_alias(text, alias)
        ]
        for alias in matched_aliases:
            diagnosis = self.diagnosis_aliases[alias]
            if any(
                alias != longer
                and alias.lower() in longer.lower()
                and self.diagnosis_aliases.get(longer) != diagnosis
                for longer in matched_aliases
            ):
                continue
            if _is_reference_only_mention(text, alias):
                continue
            polarity, confidence = self._polarity(text, alias)
            result.append(
                Observation(
                    finding=f"diagnosis:{diagnosis}",
                    source=source,
                    polarity=polarity,
                    anatomy=_extract_anatomy(text),
                    temporality=_extract_temporality(text),
                    confidence=0.98 if polarity == "positive" else max(0.9, confidence),
                    raw_text=text,
                    field_path=path,
                )
            )
        return result

    def _semantic_lab_observations(
        self,
        source: str,
        path: str,
        text: str,
    ) -> List[Observation]:
        """Normalize common binary lab semantics into reusable fact findings."""
        compact = _normalize_term(text)
        result_compact = _normalize_term(_result_value_text(text))
        context = _normalize_term(f"{source} {path} {text}")
        findings: List[Tuple[str, float]] = []
        negative = any(
            token in compact
            for token in ("阴性", "无生长", "未培养出", "未检出", "正常", "0-5", "0～5")
        )
        result_negative = any(
            token in result_compact
            for token in ("阴性", "无生长", "未培养出", "未检出", "正常", "0-5", "0～5")
        )
        result_positive = any(
            token in result_compact for token in ("阳性", "检出", "异常", "升高", "+")
        )
        positive = result_positive or any(token in compact for token in ("阳性", "检出", "异常", "升高", "+"))
        def assertive_term(terms: Iterable[str]) -> bool:
            for term in terms:
                if _normalize_term(term) not in compact:
                    continue
                polarity, _ = self._polarity(text, term)
                if polarity == "positive":
                    return True
            return False

        if "尿培养" in context or "culture" in context:
            if any(token in result_compact for token in ("阴性", "无生长", "未培养出", "未检出")):
                findings.append(("urine_culture_no_growth", 0.96))
            elif result_positive or "生长" in result_compact or (
                "cfu" in result_compact and any(ch.isdigit() for ch in result_compact)
            ) or any(
                token in result_compact
                for token in ("大肠埃希菌", "大肠杆菌", "肠球菌", "葡萄球菌", "克雷伯菌", "变形杆菌")
            ):
                findings.append(("urine_culture_positive", 0.94))
        if "ugt1a1" in context and assertive_term(("突变", "变异", "致病", "阳性", "检出")):
            findings.append(("ugt1a1_positive", 0.98))
            findings.append(("genetic_suspicion", 0.92))
        if any(token in context for token in ("鼻咽镜", "鼻咽")) and assertive_term(
            (
                "黏膜充血",
                "粘膜充血",
                "弥漫性充血",
                "充血",
                "增厚",
                "黏液",
                "痂皮",
                "红斑",
                "肿胀",
                "淋巴滤泡",
                "滤泡增生",
                "慢性炎症",
                "炎症改变",
            )
        ):
            findings.append(("nasopharyngoscopy_abnormal", 0.94))
            findings.append(("nasopharyngeal_chronic_inflammation", 0.9))
        if any(token in context for token in ("脱落细胞学", "细胞学")) and assertive_term(
            (
                "慢性炎症",
                "炎症改变",
                "炎症细胞",
                "炎性细胞浸润",
                "白细胞浸润",
                "大量脱落上皮细胞",
                "淋巴细胞",
                "中性粒细胞",
            )
        ):
            findings.append(("cytology_chronic_inflammation", 0.92))
            findings.append(("nasopharyngeal_chronic_inflammation", 0.86))
        if any(token in context for token in ("直肠指检", "dre", "前列腺")) and assertive_term(("压痛",)):
            findings.append(("prostate_tenderness", 0.96))
        if any(token in context for token in ("耳镜", "鼓膜", "耳部")):
            if assertive_term(("鼓膜充血", "鼓膜红肿", "鼓膜炎症", "鼓膜明显充血", "鼓膜红斑")):
                findings.append(("tympanic_membrane_inflammation", 0.95))
                findings.append(("acute_tympanitis", 0.9))
            if assertive_term(("鼓膜疱疹", "鼓膜大疱", "鼓膜水疱", "大疱性鼓膜炎", "鼓膜疱")):
                findings.append(("tympanic_bulla", 0.96))
                findings.append(("acute_tympanitis", 0.94))
        if any(token in context for token in ("痰培养", "真菌培养", "支气管镜", "肺泡灌洗", "病原学")):
            if assertive_term(("念珠菌", "白色念珠菌", "假丝酵母菌", "candida")):
                findings.append(("candida_positive", 0.96))
                findings.append(("fungal_pneumonia", 0.86))
        if any(token in context for token in ("裂隙灯", "眼科", "眼部", "虹膜")):
            if assertive_term(("虹膜缺损", "虹膜裂隙", "钥匙孔样瞳孔", "虹膜缺损畸形")):
                findings.append(("iris_coloboma", 0.96))
        if any(token in context for token in ("胃镜", "上消化道内镜", "内镜", "活检", "食管")):
            if assertive_term(("食管溃疡", "食道溃疡", "食管黏膜溃疡", "溃疡", "糜烂溃疡")):
                findings.append(("esophageal_ulcer", 0.96))
        if any(token in context for token in ("染色体", "核型", "基因检测", "遗传学")):
            if assertive_term(("47,xxx", "x三体", "triple x", "超雌")):
                findings.append(("triple_x_karyotype", 0.98))
                findings.append(("genetic_suspicion", 0.9))
        if any(token in context for token in ("妊娠试验", "βhcg", "β-hcg", "hcg")):
            if result_positive or any(ch.isdigit() for ch in result_compact):
                findings.append(("hcg_positive", 0.94))
                findings.append(("early_pregnancy", 0.82))
        if any(token in context for token in ("盆腔超声", "妇科超声", "阴道超声", "超声")):
            if assertive_term(("宫内孕", "宫内妊娠", "孕囊", "胎心")):
                findings.append(("early_pregnancy", 0.88))
            if assertive_term(("多囊卵巢", "多囊样卵巢", "卵巢多囊样改变")):
                findings.append(("polycystic_ovaries", 0.94))
        if any(token in context for token in ("泪腺", "眼眶", "眼部")):
            if assertive_term(("泪腺肿大", "泪腺区肿胀", "外上方肿胀", "泪腺炎")):
                findings.append(("lacrimal_gland_swelling", 0.94))
            if assertive_term(("泪腺区疼痛", "眼眶外上方疼痛", "压痛")):
                findings.append(("lacrimal_gland_pain", 0.86))
        if any(token in context for token in ("视力", "验光", "屈光", "阅读", "近距离")):
            if assertive_term(("看近模糊", "阅读困难", "近距离看不清", "老花", "老视", "屈光异常", "+")):
                findings.append(("near_vision_difficulty", 0.9))
                findings.append(("refractive_error", 0.86))
        if any(token in context for token in ("染色体", "核型", "基因", "性腺", "dsd")):
            if assertive_term(("46,XX/46,XY", "46XX/46XY", "嵌合", "DSD", "性别发育异常")):
                findings.append(("sex_development_disorder", 0.94))
                findings.append(("karyotype_mosaic", 0.9))
            if assertive_term(("卵睾", "卵巢和睾丸组织", "ovotesticular")):
                findings.append(("ovotesticular_tissue", 0.96))
        if any(token in context for token in ("颞下颌", "下颌", "tmj", "口腔颌面")):
            if assertive_term(("脱位", "关节头前移", "下颌关节脱位")):
                findings.append(("tmj_dislocation", 0.96))
            if assertive_term(("不能闭口", "闭不上嘴", "张口不能闭合")):
                findings.append(("unable_close_mouth", 0.9))
        if any(token in context for token in ("外阴", "阴道", "宫颈", "肛周", "生殖器", "hpv")):
            if assertive_term(("菜花样", "疣体", "乳头状赘生物", "尖锐湿疣", "HPV阳性")):
                findings.append(("anogenital_warts", 0.94))
                findings.append(("cauliflower_lesions", 0.9))
            if assertive_term(("滴虫", "毛滴虫", "泡沫样", "草莓样宫颈")):
                findings.append(("trichomonas_positive", 0.94))
                findings.append(("frothy_vaginal_discharge", 0.84))
            if assertive_term(("pH>4.5", "pH大于4.5", "pH升高")):
                findings.append(("vaginal_ph_high", 0.86))
        if any(token in context for token in ("脐尿管", "脐部", "脐孔", "膀胱顶部", "下腹正中")):
            if assertive_term(("脐尿管囊肿", "脐尿管残余", "囊性肿物", "膀胱顶部囊性")):
                findings.append(("urachal_cyst_imaging", 0.96))
                findings.append(("midline_suprapubic_cyst", 0.88))
            if assertive_term(("流液", "流脓", "渗液")):
                findings.append(("umbilical_discharge", 0.9))
        if "白细胞酯酶" in compact:
            findings.append(
                ("leukocyte_esterase_negative" if result_negative else "leukocyte_esterase_positive", 0.92)
            )
        if "亚硝酸盐" in compact:
            findings.append(("nitrite_negative" if result_negative else "nitrite_positive", 0.92))
        if any(token in compact for token in ("尿白细胞", "尿液白细胞")) and result_negative:
            findings.append(("urine_wbc_normal", 0.88))
        if any(token in compact for token in ("残余尿", "排尿后残余")) and result_negative:
            findings.append(("normal_postvoid_residual", 0.88))
        if not negative and any(
            token in compact
            for token in ("mpoanca", "mpo抗体", "抗mpo", "髓过氧化物酶抗体")
        ) and (positive or any(ch.isdigit() for ch in compact)):
            findings.append(("mpo_anca_positive", 0.96))
        if not negative and any(
            token in compact
            for token in ("panca", "p-anca")
        ) and (positive or any(ch.isdigit() for ch in compact)):
            findings.append(("p_anca_positive", 0.92))
        if not negative and any(
            token in compact
            for token in ("anca谱", "anca阳性", "抗中性粒细胞胞质抗体", "抗中性粒细胞胞浆抗体")
        ) and (positive or any(ch.isdigit() for ch in compact)):
            findings.append(("anca_positive", 0.9))

        return [
            Observation(
                finding=finding,
                source=source,
                polarity="positive",
                confidence=confidence,
                anatomy="尿道" if "尿道" in compact else ("膀胱" if "膀胱" in compact else ""),
                raw_text=text,
                field_path=path,
            )
            for finding, confidence in findings
        ]

    def _numeric_clinical_observations(
        self,
        source: str,
        path: str,
        text: str,
        value: Optional[float],
        unit: str,
        reference: str,
        direction: str,
        polarity: str,
    ) -> List[Observation]:
        if polarity == "negative":
            return []
        key = _normalize_term(path + " " + text)
        findings: List[Tuple[str, float]] = []
        if any(token in key for token in ("心率", "脉搏", "heartrate", "pulse", "hr")) and value is not None and value < 50:
            findings.append(("bradycardia", 0.94))
        elif any(token in key for token in ("pr间期", "printerval")) and value is not None and (
            value > 200 or value > 0.20
        ):
            findings.append(("pr_prolongation", 0.94))
        elif "镁" in key:
            findings.extend(self._magnesium_numeric_findings(key, value, direction))
        elif any(token in key for token in ("间接胆红素", "非结合胆红素", "未结合胆红素")) and direction == "high":
            findings.extend(
                [
                    ("unconjugated_hyperbilirubinemia", 0.96),
                    ("bilirubin_high", 0.92),
                ]
            )
        elif "胆红素" in key and direction == "high":
            findings.append(("bilirubin_high", 0.92))
        elif any(token in key for token in ("25羟维生素d", "25ohd", "维生素d")) and direction == "low":
            findings.append(("vitamin_d_low", 0.94))
        elif any(token in key for token in ("碱性磷酸酶", "alp")) and direction == "high":
            findings.append(("alp_elevated", 0.94))
        elif "钙" in key and "镁" not in key and direction == "low":
            findings.append(("hypocalcemia", 0.94))
        elif any(token in key for token in ("肌酐", "creatinine")) and direction == "high":
            findings.append(("renal_impairment", 0.94))
        elif any(token in key for token in ("egfr", "肾小球滤过率")) and direction == "low":
            findings.append(("egfr_low", 0.96))
        elif any(token in key for token in ("尿素氮", "bun")) and direction == "high":
            findings.append(("urea_elevated", 0.92))
        elif any(token in key for token in ("血钾", "钾")) and direction == "high":
            findings.append(("hyperkalemia", 0.9))
        elif any(token in key for token in ("血钠", "钠")) and direction == "low":
            findings.append(("hyponatremia", 0.9))
        elif any(token in key for token in ("皮质醇", "cortisol")) and direction == "low":
            findings.append(("cortisol_low", 0.94))
            findings.append(("adrenal_insufficiency", 0.86))
        elif any(token in key for token in ("acth", "促肾上腺皮质激素")) and direction == "high":
            findings.append(("acth_high", 0.92))
            findings.append(("adrenal_insufficiency", 0.86))
        elif any(token in key for token in ("孕酮", "progesterone")) and direction == "low":
            findings.append(("progesterone_low", 0.9))
        elif any(token in key for token in ("βhcg", "β-hcg", "hcg")) and (direction == "high" or (value is not None and value > 5)):
            findings.append(("hcg_positive", 0.94))
            findings.append(("early_pregnancy", 0.82))
        elif any(token in key for token in ("碳酸氢根", "hco3", "ph")) and direction == "low":
            findings.append(("metabolic_acidosis", 0.88))
        elif any(token in key for token in ("白蛋白", "albumin")) and direction == "low":
            findings.append(("hypoalbuminemia", 0.92))
        elif any(token in key for token in ("红细胞压积", "血细胞比容", "hct", "hematocrit")) and direction == "high":
            findings.append(("hemoconcentration", 0.92))
        elif any(token in key for token in ("门静脉内径", "门静脉直径", "门静脉宽度")) and value is not None and value >= 13:
            findings.append(("portal_vein_dilation", 0.92))
        elif any(token in key for token in ("血小板", "platelet", "plt")) and direction == "low":
            findings.append(("thrombocytopenia", 0.9))
        elif any(token in key for token in ("尿红细胞", "rbc")) and direction == "high":
            findings.append(("microscopic_hematuria", 0.94))
        elif any(token in key for token in ("尿白细胞", "白细胞尿", "wbc")) and direction == "high":
            findings.append(("pyuria", 0.92))
        elif any(token in key for token in ("尿细菌", "细菌计数", "bacteria")) and (
            direction == "high" or value is not None
        ):
            findings.append(("bacteriuria", 0.9))
        elif any(token in key for token in ("mpoanca", "mpo抗体", "抗mpo", "髓过氧化物酶抗体")) and (
            direction == "high" or value is not None
        ):
            findings.append(("mpo_anca_positive", 0.96))
        elif any(token in key for token in ("panca", "p-anca")) and (
            direction == "high" or value is not None
        ):
            findings.append(("p_anca_positive", 0.92))
        elif any(token in key for token in ("anca谱", "抗中性粒细胞胞质抗体", "抗中性粒细胞胞浆抗体")) and (
            direction == "high" or value is not None
        ):
            findings.append(("anca_positive", 0.9))
        elif any(token in key for token in ("肺动脉瓣压差", "峰值压差")) and value is not None and value >= 25:
            findings.append(("pulmonary_valve_gradient", 0.9))
        if not findings:
            return []
        deduped_findings = dict(findings)
        return [
            Observation(
                finding=finding,
                source=source,
                value=value,
                unit=unit,
                reference_range=reference,
                direction=direction,
                anatomy=_extract_anatomy(text),
                temporality=_extract_temporality(text),
                confidence=confidence if reference else min(confidence, 0.86),
                raw_text=text,
                field_path=path,
            )
            for finding, confidence in deduped_findings.items()
        ]

    @staticmethod
    def _magnesium_numeric_findings(
        key: str,
        value: Optional[float],
        direction: str,
    ) -> List[Tuple[str, float]]:
        is_urine = any(token in key for token in ("尿镁", "尿电解质", "尿中镁"))
        is_load_retention = "镁负荷" in key or "保留率" in key or "保留试验" in key
        findings: List[Tuple[str, float]] = []

        if is_load_retention and (direction == "high" or (value is not None and value > 30)):
            findings.extend(
                [
                    ("magnesium_load_retention_high", 0.96),
                    ("magnesium_depletion", 0.95),
                ]
            )
            return findings
        if direction != "low":
            return findings
        if is_urine:
            findings.extend(
                [
                    ("low_urine_magnesium", 0.9),
                    ("magnesium_depletion", 0.9),
                ]
            )
        else:
            findings.append(("low_magnesium", 0.96))
        return findings

    @staticmethod
    def _polarity(text: str, term: str = "") -> Tuple[str, float]:
        target = str(text or "")
        if term:
            idx = target.lower().find(term.lower())
            if idx >= 0:
                # Limit scope to the clause containing the finding. This keeps
                # "未见 ASD，重度二尖瓣反流" from negating the valve finding,
                # while still treating parenthetical examples in
                # "未见先天性缺损（如 ASD、VSD）" as negative.
                left = max(
                    target.rfind(mark, 0, idx)
                    for mark in ("，", ",", "。", ";", "；", "！", "!", "？", "?", "\n")
                )
                right_positions = [
                    target.find(mark, idx + len(term))
                    for mark in ("，", ",", "。", ";", "；", "！", "!", "？", "?", "\n")
                ]
                right_positions = [pos for pos in right_positions if pos >= 0]
                right = min(right_positions) if right_positions else len(target)
                clause = target[left + 1:right]
                if _NEGATION_RE.search(clause):
                    return "negative", 0.94
                if _UNCERTAINTY_RE.search(clause):
                    return "uncertain", 0.6
                return "positive", 0.86 if _POSITIVE_RE.search(clause) else 0.78
        if _NEGATION_RE.search(target):
            # A leaf is a narrow semantic field. A leading negative applies to
            # examples inside the same value, including "未见...(如 ASD)".
            first_negative = _NEGATION_RE.search(target)
            first_positive = _POSITIVE_RE.search(target)
            if first_negative and (not first_positive or first_negative.start() <= first_positive.start()):
                return "negative", 0.92
        if _UNCERTAINTY_RE.search(target):
            return "uncertain", 0.6
        return "positive", 0.86 if _POSITIVE_RE.search(target) else 0.78

    @staticmethod
    def _numeric_details(text: str) -> Tuple[Optional[float], str, str, str]:
        result_text = _result_value_text(text)
        value_area, reference = _split_measurement_and_reference(result_text)
        value_match = re.search(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)", value_area)
        value = float(value_match.group(1)) if value_match else None
        unit = ""
        if value_match:
            tail = value_area[value_match.end():value_match.end() + 32]
            unit_match = re.match(r"\s*([^\s，,；;。()\[\]［］]+)", tail)
            if unit_match:
                unit = unit_match.group(1)

        direction = _direction_from_reference(value, reference)
        if not direction:
            direction = _direction_from_words(text)
        return value, unit, reference, direction

    def _load_diagnosis_aliases(self) -> Dict[str, str]:
        aliases: Dict[str, str] = {}
        catalog_path = os.path.join(self.ref_dir, "diseases_catalog.json")
        for item in _read_json(catalog_path, {}).get("diseases", []):
            name = str(item.get("name") or "").strip()
            if name:
                aliases[name] = name

        if os.path.isdir(self.ref_dir):
            for filename in sorted(os.listdir(self.ref_dir)):
                if not filename.startswith("disease_profiles") or not filename.endswith(".json"):
                    continue
                data = _read_json(os.path.join(self.ref_dir, filename), {})
                for profile in data.get("profiles", []):
                    name = str(profile.get("name") or "").strip()
                    if not name:
                        continue
                    aliases[name] = name
                    for alias in profile.get("aliases", []) or []:
                        if str(alias).strip():
                            aliases[str(alias).strip()] = name

        extension_path = os.path.join(self.ref_dir, "submission_diagnosis_extensions.json")
        for item in _read_json(extension_path, {}).get("extensions", []):
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            aliases[name] = name
            for alias in item.get("aliases", []) or []:
                if str(alias).strip():
                    aliases[str(alias).strip()] = name
        # Prefer longer terms so a precise diagnosis is emitted before a parent term.
        return dict(sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True))

    @staticmethod
    def _dedupe(items: Sequence[Observation]) -> List[Observation]:
        best: Dict[Tuple[str, str, str, str], Observation] = {}
        for item in items:
            key = (item.finding, item.source, item.polarity, item.field_path)
            current = best.get(key)
            if current is None or item.confidence > current.confidence:
                best[key] = item
        return list(best.values())


def _flatten_leaves(value: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_leaves(item, path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _flatten_leaves(item, path)
        return
    yield prefix or "value", value


def _as_text_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_term(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[\s，。！？；：、,.!?;:|/\\()（）【】\[\]{}_-]+", "", text)


def _result_value_text(text: str) -> str:
    target = str(text or "").strip()
    for sep in (":", "："):
        index = target.find(sep)
        if index < 0:
            continue
        label = target[:index]
        if sep == "：" and re.search(r"(?:参考|正常范围|正常值)", label):
            continue
        return target[index + 1:].strip()
    return target


def _split_measurement_and_reference(text: str) -> Tuple[str, str]:
    target = str(text or "").strip()
    ref_match = re.search(r"(?:参考值|参考范围|参考区间|正常范围|正常值|参考)", target)
    if not ref_match:
        return target, ""
    value_area = target[:ref_match.start()].strip(" ［[]（(；;，,。")
    reference = target[ref_match.start():].strip(" ］]）)")
    return value_area, reference


def _direction_from_reference(value: Optional[float], reference: str) -> str:
    if value is None or not reference:
        return ""
    ref = (
        str(reference)
        .replace("＜", "<")
        .replace("≤", "<=")
        .replace("≦", "<=")
        .replace("＞", ">")
        .replace("≥", ">=")
        .replace("≧", ">=")
    )

    upper_match = re.search(
        r"(?:<=|<|小于|低于|不超过|少于)\s*(-?\d+(?:\.\d+)?)"
        r"(?:\s*[-~～—至到]\s*(-?\d+(?:\.\d+)?))?",
        ref,
        flags=re.IGNORECASE,
    )
    if upper_match:
        upper = float(upper_match.group(2) or upper_match.group(1))
        return "high" if value > upper else "normal"

    lower_match = re.search(
        r"(?:>=|>|大于|高于|不少于|至少)\s*(-?\d+(?:\.\d+)?)"
        r"(?:\s*[-~～—至到]\s*(-?\d+(?:\.\d+)?))?",
        ref,
        flags=re.IGNORECASE,
    )
    if lower_match:
        lower = float(lower_match.group(1))
        return "low" if value < lower else "normal"

    range_match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*(?:[-~～—至到]\s*(-?\d+(?:\.\d+)?))",
        ref,
        flags=re.IGNORECASE,
    )
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        if value < low:
            return "low"
        if value > high:
            return "high"
        return "normal"

    single_match = re.search(r"(-?\d+(?:\.\d+)?)", ref, flags=re.IGNORECASE)
    if single_match and any(token in ref for token in ("上限", "高限")):
        return "high" if value > float(single_match.group(1)) else "normal"
    if single_match and any(token in ref for token in ("下限", "低限")):
        return "low" if value < float(single_match.group(1)) else "normal"
    return ""


def _direction_from_words(text: str) -> str:
    target = str(text or "")
    if any(token in target for token in ("降低", "偏低", "减低", "低于参考", "低于正常", "低于")):
        return "low"
    if any(token in target for token in ("升高", "偏高", "增高", "高于参考", "高于正常", "高于")):
        return "high"
    return ""


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def _contains_alias(text: str, alias: str) -> bool:
    if not alias:
        return False
    if alias.isascii() and len(alias) <= 6:
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
        )
    return alias.lower() in text.lower()


def _is_reference_only_mention(text: str, term: str) -> bool:
    """Return true when a term appears only in a reference/example label."""
    target = str(text or "")
    index = target.lower().find(str(term or "").lower())
    if index < 0:
        return False
    boundary = max(
        target.rfind(mark, 0, index)
        for mark in ("，", ",", "。", ";", "；", "！", "!", "？", "?", "\n")
    )
    prefix = target[boundary + 1:index]
    has_reference_marker = bool(
        re.search(r"(?:参考值|参考范围|正常范围|示例|例如|举例|术语说明)\s*[：:]?", prefix)
    )
    if not has_reference_marker:
        return False
    # A real assertion such as "参考说明：超声提示 ASD" is still evidence;
    # bare catalog/example text is ignored.
    return not (_NEGATION_RE.search(prefix) or _POSITIVE_RE.search(prefix))


def _extract_anatomy(text: str) -> str:
    return next((term for term in _ANATOMY_TERMS if term in str(text or "")), "")


def _extract_temporality(text: str) -> str:
    target = str(text or "")
    match = re.search(
        r"(?:近|约|持续)?\s*\d+(?:\.\d+)?\s*(?:小时|天|日|周|月|年)(?:前|来|余)?",
        target,
    )
    if match:
        return match.group(0).strip()
    return next(
        (term for term in ("急性", "亚急性", "慢性", "反复", "进行性", "突发") if term in target),
        "",
    )


def _is_history_path(path: str) -> bool:
    compact = _normalize_term(path)
    return any(
        token in compact
        for token in ("history", "past", "family", "allerg", "medication", "既往", "家族", "过敏", "用药", "病史")
    )


def _is_physical_source(source: str) -> bool:
    return any(token in str(source or "") for token in ("体格", "查体", "体检", "Physical"))


def _is_lab_source(source: str) -> bool:
    return any(
        token in str(source or "")
        for token in (
            "血", "尿", "肾功能", "肝功能", "电解质", "CBC", "CRP", "培养", "凝血",
            "白蛋白", "肌酐", "尿素", "eGFR", "激素", "代谢", "生化",
        )
    )


def _is_imaging_source(source: str) -> bool:
    return any(
        token in str(source or "")
        for token in ("超声", "CT", "X线", "MRI", "影像", "心导管", "造影", "CXR", "CMR")
    )


def _is_risk_factor(item: Observation) -> bool:
    text = f"{item.finding} {item.raw_text}"
    return any(
        token in text
        for token in (
            "促排", "取卵", "辅助生殖", "试管婴儿", "hCG", "肝硬化", "慢性肾",
            "透析", "先天", "家族史", "长期", "免疫抑制",
        )
    )


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return default
