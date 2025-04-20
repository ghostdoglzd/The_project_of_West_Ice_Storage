import torch
import faiss
import random
import re
import numpy as np
from collections import defaultdict
from transformers import BertTokenizer, BertForTokenClassification
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import logging
from transformers import logging as transformers_logging

# 设置transformers库的日志级别为ERROR，避免输出警告
transformers_logging.set_verbosity_error()

# 初始化核心组件
ner_tokenizer = BertTokenizer.from_pretrained("./model/dslim/bert-base-NER")
ner_model = BertForTokenClassification.from_pretrained("./model/dslim/bert-base-NER").eval()
semantic_model = SentenceTransformer('./model/all-MiniLM-L6-v2').eval()

def preprocess_text(text, max_length=512):
    """智能分块处理"""
    sentences = re.findall(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s*[^.!?]+', text)
    chunks, current_chunk = [], []
    current_len = 0
    
    for sent in sentences:
        sent_tokens = ner_tokenizer.tokenize(sent)
        if current_len + len(sent_tokens) > max_length - 2:
            chunks.append(' '.join(current_chunk))
            current_chunk, current_len = [], 0
        current_chunk.append(sent)
        current_len += len(sent_tokens)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def extract_entities(text):
    """改进的实体提取"""
    chunks = preprocess_text(text)
    all_entities = []
    
    for chunk in chunks:
        inputs = ner_tokenizer(chunk, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = ner_model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)[0]
        tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tags = [ner_model.config.id2label[p] for p in predictions.tolist()]
        
        current_entity = []
        current_tag = None
        for token, tag in zip(tokens[1:-1], tags[1:-1]):  # 跳过[CLS]和[SEP]
            if tag.startswith("B-"):
                if current_entity:
                    all_entities.append((' '.join(current_entity), current_tag))
                current_entity = [token]
                current_tag = tag[2:]
            elif tag.startswith("I-") and current_tag == tag[2:]:
                current_entity.append(token)
            else:
                if current_entity:
                    all_entities.append((' '.join(current_entity), current_tag))
                current_entity = []
                current_tag = None
                
    return list(set([(ent.replace(" ##", ""), tag) for ent, tag in all_entities]))  # 清理合并的token

# 风险关键词列表（同前）
risk_terms = [
    # Technology Company-Specific Risks (新增分类)
    "Huawei 5G technology sanctions", "Huawei supply chain decoupling", "Huawei HarmonyOS ecosystem fragmentation",
    "Xiaomi smartphone market volatility", "Xiaomi overseas data compliance issues", "Xiaomi IoT device security breach",
    "Alibaba cloud service outage", "Alibaba cross-border data flow restrictions", "Alibaba antitrust compliance failure",
    "Tencent gaming content regulation", "Tencent social media misinformation crisis", "Tencent fintech system vulnerability",
    "Baidu autonomous driving liability dispute", "Baidu AI ethics algorithm bias", "Baidu Apollo platform security flaw",
    "TikTok content moderation backlash", "TikTok global privacy litigation", "TikTok algorithmic transparency crisis",
    "ByteDance cross-border content censorship", "ByteDance AI training data privacy breach", "ByteDance global regulatory divergence",
    "ZTE US export restrictions renewal", "ZTE telecom equipment security audit failure", "ZTE 5G infrastructure dependency risk",
    "JD.com logistics automation failure", "JD.com supply chain AI prediction error", "JD.com counterfeit product detection crisis",
    "Meituan delivery algorithm abuse scandal", "Meituan local service data monopoly", "Meituan cross-industry competition backlash",
    "DJI drone export control escalation", "DJI aviation safety certification failure", "DJI global market localization conflict",
    "Bytedance TikTok-ByteDance corporate governance risk", "Huawei-Hisilicon chip design IP dispute", "Xiaomi-POCO brand localization backlash",
    "Chinese tech companies US listing restrictions", "Chinese AI companies data localization conflict", "Chinese cloud providers cross-border audit crisis",
    "Global semiconductor foundry dependency risk", "Critical technology IP litigation", "Cross-border technology transfer disputes",
    "AI company algorithmic transparency mandates", "Tech giants antitrust breakup threat", "Cloud infrastructure geopolitical risk",
    "Consumer electronics brand reputation crisis", "Tech company carbon neutrality compliance failure", "Open-source software license litigation"

    # 货币政策与金融市场动荡
    "interest rate hike", "quantitative tightening", "bond yield surge", "sovereign debt crisis",
    "margin call", "credit rating downgrade", "repo market freeze", "junk bond collapse",
    "hyperinflation", "stagflation", 
    "crypto market systemic collapse", "currency war escalation", "capital control contagion", 
    "central bank digital currency (CBDC) malfunction", "algorithmic stablecoin depeg", 
    "global interest rate policy divergence", "cross-border capital flow reversal", 
    "shadow banking liquidity crunch", "sovereign wealth fund solvency crisis", 
    "emerging market currency peg collapse", "debt-for-equity swap backlash"  # 新增5项

    # 行业性黑天鹅事件
    "semiconductor shortage", "rare earth embargo", "pharma trial failure", "EV battery recall",
    "airline bankruptcy", "shipping container backlog", "real estate bubble", "crypto exchange collapse",
    "AI ethics scandal", 
    "quantum computing security breach", "autonomous vehicle system failure", "gene editing mishap", 
    "metaverse platform collapse", "cloud computing outage", "critical mineral cartel formation", 
    "biometric authentication fraud", "nanotechnology safety crisis", "space debris collision cascade", 
    "agricultural drone swarm malfunction", "synthetic biology containment breach"  # 新增5项

    # 自然灾害与气候风险
    "extreme weather lockdown", "crop failure", "water scarcity", "carbon tax riot",
    "hurricane disruption", "wildfire insurance crisis", "flood supply chain", "El Niño commodity shock",
    "permafrost thaw acceleration", "coastal city submersion", "carbon capture failure", 
    "climate litigation surge", "megadrought", "volcanic ash supply chain collapse", 
    "global coral reef collapse", "ozone layer recovery reversal", "atmospheric methane spike", 
    "arctic permafrost methane release", "hydropower reservoir sedimentation crisis"  # 新增5项

    # 公共卫生危机
    "pandemic resurgence", "vaccine side effect panic", "antibiotic resistance", "hospital overload",
    "WHO emergency alert", "lab leak controversy", "zoonotic spillover",
    "unknown pathogen outbreak", "mental health pandemic", "antiviral resistance crisis", 
    "medical AI misdiagnosis wave", "global blood shortage", "vaccine nationalism escalation", 
    "gene therapy adverse reaction crisis", "organ shortage black market surge", 
    "elderly care system collapse", "healthcare AI liability dispute", "pandemic vaccine inequity riot"  # 新增5项

    # 法律与监管突变
    "antitrust breakup", "data privacy fine", "ESG compliance crash", "insider trading probe",
    "IPO suspension", "short selling ban", "derivative regulation", "cross-border audit conflict",
    "AI governance framework clash", "data sovereignty dispute", "quantum encryption regulation", 
    "ESG greenwashing crackdown", "AI liability law ambiguity", "offshore tax haven collapse", 
    "cross-border data localization mandate", "AI facial recognition ban", 
    "quantum computing IP theft crackdown", "blockchain smart contract litigation surge", 
    "offshore financial center collapse"  # 新增5项

    # 供应链与劳动力风险
    "port strike", "chip fab fire", "battery plant protest", "mineral export ban",
    "trucker shortage", "union wage demand", "cross-strait logistics freeze", "child labor lawsuit",
    "supply chain decoupling", "AI labor displacement", "critical skill shortage", 
    "3D printing material crisis", "agricultural labor automation backlash", "cross-border labor ban", 
    "cross-border robotics tariff war", "labour force aging crisis", 
    "autonomous logistics system failure", "critical infrastructure staffing shortage", 
    "global supply chain carbon footprint audit"  # 新增5项

    # 能源与材料危机
    "OPEC+ production cut", "LNG terminal explosion", "uranium shortage", "cobalt mining ban",
    "graphite embargo", "lithium cartel", "hydrogen leak panic",
    "solar panel supply chain fracture", "rare earth recycling crisis", "critical mineral embargo", 
    "nuclear waste management failure", "biofuel feedstock shortage", "hydrogen infrastructure collapse", 
    "geothermal energy seismic risk", "uranium enrichment facility sabotage", 
    "critical battery material price spike", "solar panel silicon shortage", 
    "ocean thermal energy conflict"  # 新增5项

    # 行为金融与市场情绪
    "panic selling", "FOMO crash", "margin debt unwind", "whale account liquidation",
    "dark pool manipulation", "gamma squeeze", "rumor-driven selloff",
    "social media-driven volatility", "AI trading algorithm meltdown", "central bank communication crisis", 
    "currency carry trade collapse", "ETF liquidity crunch", "decentralized finance (DeFi) flash crash", 
    "AI-generated market manipulation", "blockchain oracle data fraud", 
    "central bank communication credibility crisis", "algorithmic trading liquidity void", 
    "market sentiment sentiment analysis distortion"  # 新增5项

    # 地缘博弈升级
    "Taiwan Strait militarization", "South China Sea collision", "Arctic resource clash",
    "Mekong River dam war", "Kashmir border skirmish", "Sudan proxy war", "Nagorno-Karabakh escalation",
    "Arctic shipping route control", "space resource claim conflict", "5G spectrum war", 
    "cyber warfare escalation", "critical infrastructure sabotage", "global satellite navigation jamming", 
    "cross-border data flow war", "critical mineral supply chain weaponization", "quantum supremacy arms race", 
    "polar ice cap territorial dispute", "deep-sea mining conflict", 
    "space debris weaponization", "AI military system arms race"  # 新增4项（保持总数平衡）
]

# 行业词库（同前）
industry_terms = {
    # 基础材料
    "金属采矿": [
        "Iron ore mining rights", "Copper concentrate processing", "Bauxite exploration",
        "Rare earth refining", "Tungsten carbide production"
    ],
    "工业化学品": [
        "Polyethylene synthesis", "Chlor-alkali production units", "Titanium dioxide pigments",
        "Industrial solvent purification", "Synthetic resin polymerization"
    ],
    "特种化工": [
        "Carbon fiber precursors", "High-purity alumina", "Electronic-grade hydrofluoric acid",
        "Specialty ceramic coatings", "Industrial gas separation"
    ],
    "林业与纸制品": [
        "Kraft pulp production", "Corrugated packaging materials", "FSC-certified forestry",
        "Recycled paperboard manufacturing", "Laminated wood panels"
    ],
    "化肥与农业资源": [
        "Nitrogen fertilizer synthesis", "Phosphate rock processing", "Potash mining technology",
        "Soil amendment formulations", "Micronutrient fertilizers"
    ],
    "稀土与稀有金属": [
        "Neodymium-praseodymium alloy", "Cobalt concentrate refining", "Tantalum capacitor materials",
        "Rare earth permanent magnets", "Tungsten carbide powder"
    ],

    # 能源
    "石油勘探": [
        "Shale gas fracking", "Offshore drilling rigs", "Reservoir simulation software",
        "Seismic imaging technology", "Oil sands extraction"
    ],
    "炼油与分销": [
        "Catalytic cracking units", "Gasoline blending optimization", "Pipeline corrosion monitoring",
        "Bunker fuel specifications", "Jet fuel additives"
    ],
    "油气服务": [
        "Hydraulic fracturing pumps", "Subsea Christmas trees", "Drilling mud systems",
        "Oilfield cementing services", "Pipeline integrity management"
    ],
    "液化天然气": [
        "Floating storage regasification", "Cryogenic tank insulation", "Boil-off gas recovery",
        "Small-scale LNG terminals", "Bunkering infrastructure"
    ],
    "太阳能设备": [
        "PERC solar cells", "Bifacial module technology", "Solar tracking algorithms",
        "Anti-reflective coatings", "PV inverter efficiency"
    ],
    "风能开发": [
        "Offshore wind turbine foundations", "Blade pitch control systems", "Nacelle yaw mechanisms",
        "Grid connection converters", "Wind resource assessment"
    ],
    "氢能源技术": [
        "Proton exchange membrane electrolyzers", "Hydrogen compression stations", "Fuel cell stacks",
        "Ammonia cracking reactors", "Hydrogen refueling protocols"
    ],
    "生物燃料": [
        "Algae biofuel cultivation", "Cellulosic ethanol fermentation", "Biodiesel transesterification",
        "Pyrolysis oil upgrading", "Anaerobic digestion plants"
    ],

    # 工业
    "航空航天": [
        "Composite fuselage manufacturing", "Turbofan engine blades", "Satellite attitude control",
        "Radar-absorbing materials", "Aircraft maintenance MRO"
    ],
    "铁路设备": [
        "Bogie frame casting", "Pantograph current collection", "Axle load monitoring",
        "Container intermodal systems", "Railway signaling ETCS"
    ],
    "工业机械": [
        "CNC machining centers", "Industrial robot end-effectors", "Hydraulic press automation",
        "Laser cutting optics", "Additive manufacturing powders"
    ],
    "建筑工程": [
        "BIM coordination software", "Prefabricated concrete elements", "Geotechnical instrumentation",
        "Construction waste recycling", "Smart building IoT"
    ],
    "电气设备": [
        "Transformer core lamination", "Smart grid sensors", "High-voltage circuit breakers",
        "Energy storage BMS", "Wireless power transfer"
    ],
    "物流与货运": [
        "Cross-docking optimization", "Cold chain logistics", "Autonomous warehouse robots",
        "Freight rate benchmarking", "Last-mile delivery drones"
    ],
    "无人机技术": [
        "Multispectral crop imaging", "LiDAR terrain mapping", "Swarm coordination algorithms",
        "BVLOS operations", "Payload release mechanisms"
    ],
    "3D打印技术": [
        "Selective laser sintering", "Digital light processing", "Binder jetting systems",
        "Topology optimization software", "Post-processing finishing"
    ],

    # 金融
    "商业银行": [
        "Collateralized loan obligations", "Basel III compliance", "Cross-border payment rails",
        "Digital onboarding platforms", "Credit risk scoring models"
    ],
    "投资银行": [
        "SPAC merger advisory", "Algorithmic trading engines", "Dark pool liquidity",
        "ESG bond underwriting", "Prime brokerage services"
    ],
    "保险服务": [
        "Actuarial mortality tables", "Catastrophe bond issuance", "Telematics-based pricing",
        "Parametric insurance triggers", "Blockchain claims processing"
    ],
    "资产管理": [
        "Smart beta ETFs", "Private equity co-investment", "Quantamental strategies",
        "FX hedging overlays", "Liquidity stress testing"
    ],
    "金融科技": [
        "Central bank digital currency", "DeFi yield farming", "Cross-chain atomic swaps",
        "Contactless payment terminals", "AI fraud detection"
    ],
    "消费信贷": [
        "Buy now pay later (BNPL)", "Credit line management", "Alternative credit scoring",
        "Debt consolidation loans", "Dynamic credit limits"
    ],
    "房地产信托": [
        "Net lease agreements", "Capitalization rate analysis", "Tenant improvement allowances",
        "Green building certifications", "REIT dividend yield"
    ],

    # 信息技术
    "半导体": [
        "Wafer fabrication", "EDA software", "Advanced packaging", "Photolithography systems",
        "Chiplet technology"
    ],
    "云计算": [
        "Hybrid cloud solutions", "Serverless computing", "Edge computing nodes",
        "Cloud security protocols", "Multi-cloud management"
    ],
    "人工智能": [
        "Neural network training", "Natural language processing (NLP)", "Computer vision algorithms",
        "Reinforcement learning frameworks", "AI ethics compliance"
    ],
    "网络安全": [
        "Zero-trust architecture", "Endpoint detection and response (EDR)", "Threat intelligence platforms",
        "Penetration testing tools", "Quantum encryption"
    ],
    "企业软件": [
        "ERP system integration", "CRM automation", "Low-code development platforms",
        "Business intelligence dashboards", "SaaS subscription models"
    ],
    "5G通信": [
        "Massive MIMO antennas", "Network slicing configurations", "mmWave spectrum allocation",
        "Open RAN architecture", "5G core networks"
    ],
    "量子计算": [
        "Qubit stabilization", "Quantum error correction", "Quantum annealing systems",
        "Photonic quantum chips", "Quantum algorithm development"
    ],
    "光电子技术": [
        "Optical transceivers", "LiDAR sensors", "Photovoltaic cells", "Fiber Bragg gratings",
        "Optoelectronic integrated circuits"
    ]
}

class RiskAnalyzer:
    def __init__(self, risk_terms):
        self.risk_index = faiss.IndexFlatIP(384)
        self._build_risk_index(risk_terms)
        
    def _build_risk_index(self, terms):
        embeddings = semantic_model.encode(terms, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self.risk_index.add(embeddings)
        self.risk_terms = terms
        
    @lru_cache(maxsize=1000)
    def analyze(self, text, keyword_threshold=0.6, top_n=5):
        # 精确匹配
        exact_matches = [term for term in self.risk_terms 
                        if re.search(rf'\b{re.escape(term)}\b', text, re.I)]
        
        # 语义匹配
        query_emb = semantic_model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        _, indices = self.risk_index.search(query_emb, top_n)
        semantic_matches = [self.risk_terms[i] for i in indices[0]]
        
        # 准备风险提示模板
        risk_templates = [
            "监测到'{match}'相关风险，可能对市场造成冲击。",
            "请注意，'{match}'可能引发行业波动，建议谨慎评估。",
            "风险提示：'{match}'或导致投资环境变化，请保持关注。",
            "检测到潜在风险因素'{match}'，可能影响相关资产表现。",
            "市场风险预警：'{match}'可能带来不确定性。",
            "投资者需警惕'{match}'的潜在负面影响。",
            "行业观察：'{match}'风险升高，建议调整策略。",
            "重要风险提示：'{match}'可能成为市场变数。"
        ]
        
        # 去重并生成多样化提示
        matches = list(set(exact_matches + semantic_matches))
        random.shuffle(risk_templates)  # 随机打乱模板顺序
        
        enriched_matches = []
        for i, match in enumerate(matches[:top_n]):
            template = risk_templates[i % len(risk_templates)]
            enriched_matches.append(template.format(match=match))
        
        return enriched_matches[:top_n]

class IndustryMatcher:
    def __init__(self, industry_terms, semantic_threshold=0.65):
        self.industry_terms = industry_terms
        self.threshold = semantic_threshold
        self._precompute_embeddings()
        
    def _precompute_embeddings(self):
        self.industry_descs = {
            ind: ' '.join(terms) for ind, terms in self.industry_terms.items()
        }
        embeddings = semantic_model.encode(list(self.industry_descs.values()), 
                                         convert_to_numpy=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.industry_list = list(self.industry_descs.keys())
    
    def _keyword_match(self, entities):
        scores = defaultdict(int)
        for entity, ent_type in entities:
            if ent_type not in ['ORG', 'PRODUCT', 'LOC']: 
                continue
            for industry, terms in self.industry_terms.items():
                if any(entity in term for term in terms):
                    scores[industry] += 1
        return scores
    
    def match(self, text, entities, top_n=3):
        # 关键词匹配
        keyword_scores = self._keyword_match(entities)
        if keyword_scores:
            sorted_industries = sorted(keyword_scores.items(), 
                                      key=lambda x: x[1], reverse=True)
            industries = [ind for ind, _ in sorted_industries[:top_n]]
        else:
            query_emb = semantic_model.encode([text], convert_to_numpy=True)
            faiss.normalize_L2(query_emb)
            _, indices = self.index.search(query_emb, top_n)
            industries = [self.industry_list[i] for i in indices[0]]
        
        # 准备行业推荐模板
        industry_templates = [
            "推荐关注行业：'{industry}'，近期发展潜力较大。",
            "行业动态：'{industry}'领域或存在投资机会。",
            "建议重点关注'{industry}'行业的市场动向。",
            "投资机会：'{industry}'领域可能迎来增长期。",
            "行业分析：'{industry}'具备较高的关注价值。",
            "'{industry}'行业前景广阔，建议纳入观察列表。",
            "市场趋势显示，'{industry}'行业值得投资者留意。",
            "关注领域推荐：'{industry}'或成为下一热点。"
        ]
        
        # 生成多样化推荐语
        random.shuffle(industry_templates)  # 随机打乱模板顺序
        enriched_industries = []
        for i, industry in enumerate(industries[:top_n]):
            template = industry_templates[i % len(industry_templates)]
            enriched_industries.append(template.format(industry=industry))
        
        return enriched_industries
def generate_report(text, risk_top_n=5, industry_top_n=3):
    """综合报告生成（更新版）"""
    # 风险分析
    risk_analyzer = RiskAnalyzer(risk_terms)
    risks = risk_analyzer.analyze(text, top_n=risk_top_n)
    
    # 行业匹配
    entities = extract_entities(text)
    industry_matcher = IndustryMatcher(industry_terms)
    industries = industry_matcher.match(text, entities, top_n=industry_top_n)
    
    # 定义报告头部（新增部分）
    risk_header = random.choice([
        "【风险预警雷达】检测到以下投资风险因素：",
        "【风险扫描报告】发现潜在风险指标：",
        "【风控提示】需要关注的风险信号：",
        "【风险扫描结果】识别到以下风险要素："
    ])
    
    industry_header = random.choice([
        "【行业机遇发现】推荐关注领域：",
        "【投资机会扫描】潜在机会行业：",
        "【行业趋势洞察】建议关注方向：",
        "【板块机会提示】值得留意领域："
    ])

    # 提取关键词
    keywords = set()
    for risk in risks:
        matches = re.findall(r"'([^']+)'", risk)
        keywords.update(matches)
    for industry in industries:
        matches = re.findall(r"'([^']+)'", industry)
        keywords.update(matches)
    
    # 组合原始报告格式
    risk_part = f"{risk_header}\n" + ("\n".join(risks) if risks else "▶ 当前文本未检测到显著风险因素")
    industry_part = f"\n\n{industry_header}\n" + ("\n".join(industries) if industries else "▶ 当前文本未匹配到特定行业")
    
    return {
        "keywords": list(keywords),
        "report": f"{risk_part}{industry_part}"
    }

if __name__ == "__main__":
    text = """
    Apple announces new iPhone 16 with revolutionary AI features. NASA discovers signs of water on Mars, sparking new exploration plans. Global stock markets tumble as oil prices hit record highs. Elon Musk's Tesla unveils affordable electric pickup truck. Celebrity couple splits after 10 years of marriage, sources say. Major cyberattack hits government systems worldwide, causing chaos. Climate change summit in Paris ends with historic agreement. New study finds link between social media use and mental health decline. Olympic gold medalist tests positive for banned substance. Bitcoin price surges past $100,000 amid regulatory uncertainty. Controversial new law passed in European Union, facing backlash. Huge asteroid narrowly misses Earth, scientists relieved. Celebrated author releases long-awaited new novel. Major Hollywood studio announces all-female superhero film. Pandemic variant resurges, prompting new travel restrictions. Robotics company unveils humanoid assistant for home use. Political scandal rocks Capitol Hill, hearings begin. Record-breaking heatwave hits North America, breaking records. New wonder drug approved for treating Alzheimer's. SpaceX launches first all-civilian mission to the Moon. Major fast-food chain introduces plant-based burger. Fashion icon's latest collection goes viral on social media. Controversial video game banned in multiple countries. Scientists discover new species in Amazon rainforest. Global protests erupt over proposed internet censorship laws.
    """

    output = generate_report(text)
    print(output)