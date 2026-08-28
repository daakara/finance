/**
 * Canonical Single Source of Truth for Company Metadata, Moats, Risk Profiles, and Sector Catalysts.
 * Eliminates duplicate dictionaries across Terminal, Compare, Sizer, and API layers.
 */

export interface AssetCatalystProfile {
  trial: string;
  phase: string;
  timeline: string;
  thesis: string;
}

export const CANONICAL_ASSET_NAMES: Record<string, string> = {
  NVDA: 'NVIDIA Corporation',
  AAPL: 'Apple Inc.',
  MSFT: 'Microsoft Corporation',
  TSLA: 'Tesla Inc.',
  PLTR: 'Palantir Technologies Inc.',
  NVO: 'Novo Nordisk A/S',
  LLY: 'Eli Lilly & Company',
  SPY: 'SPDR S&P 500 ETF Trust',
  QQQ: 'Invesco QQQ Trust',
  SMH: 'VanEck Semiconductor ETF',
  IWM: 'iShares Russell 2000 ETF',
  GLD: 'SPDR Gold Shares',
  TLT: 'iShares 20+ Year Treasury Bond ETF',
  XLE: 'Energy Select Sector SPDR Fund',
  XLK: 'Technology Select Sector SPDR Fund',
  CPRX: 'Catalyst Pharmaceuticals, Inc.',
  POWI: 'Power Integrations, Inc.',
  LNTH: 'Lantheus Holdings, Inc.',
  KO: 'The Coca-Cola Company',
  SBUX: 'Starbucks Corporation',
  AMZN: 'Amazon.com, Inc.',
  GOOGL: 'Alphabet Inc.',
  AMD: 'Advanced Micro Devices, Inc.',
  ARM: 'Arm Holdings plc',
  SMCI: 'Super Micro Computer, Inc.',
  COIN: 'Coinbase Global, Inc.',
  VRT: 'Vertiv Holdings Co',
  ISRG: 'Intuitive Surgical, Inc.',
  KLAC: 'KLA Corporation',
  CIEN: 'Ciena Corporation',
  ACLS: 'Axcelis Technologies, Inc.',
  TMDX: 'TransMedics Group, Inc.',
  MEDP: 'Medpace Holdings, Inc.',
  ELF: 'e.l.f. Beauty, Inc.',
  DUOL: 'Duolingo, Inc.',
  JPM: 'JPMorgan Chase & Co.',
  V: 'Visa Inc.',
  MA: 'Mastercard Incorporated',
  DIS: 'The Walt Disney Company',
  COST: 'Costco Wholesale Corporation',
  WMT: 'Walmart Inc.',
  CRWD: 'CrowdStrike Holdings, Inc.',
  PANW: 'Palo Alto Networks, Inc.',
  MSTR: 'MicroStrategy Incorporated',
  MARA: 'MARA Holdings, Inc.',
  IONQ: 'IonQ, Inc.',
  RKLB: 'Rocket Lab USA, Inc.',
  VRTX: 'Vertex Pharmaceuticals Incorporated',
  ETN: 'Eaton Corporation plc',
  ANET: 'Arista Networks, Inc.',
};

export const CANONICAL_ASSET_MOATS: Record<string, string> = {
  NVDA: 'Dominant proprietary CUDA computing ecosystem and NVLink interconnect switches establishing near-monopolistic datacenter AI infrastructure moats.',
  AAPL: '2.2B active installed hardware device ecosystem generating high-margin recurring services and unmatched consumer brand retention.',
  MSFT: 'Mission-critical commercial enterprise cloud infrastructure (Azure) and enterprise productivity software moats.',
  TSLA: 'Vertically integrated EV manufacturing standard and proprietary end-to-end vision neural network autonomous fleet.',
  PLTR: 'Defense-grade ontological operating platform (AIP) with IL6/JWCC security clearances for mission-critical government & commercial deployments.',
  NVO: 'Secular obesity and diabetes therapeutic franchise with accelerating multi-billion manufacturing capacity and oral Amycretin pipelines.',
  LLY: 'Global pharmaceutical market cap leader driving triple-agonist metabolic platforms (Retatrutide) and oral GLP-1 therapies (Orforglipron).',
  SPY: 'Direct cap-weighted diversification across 500 leading US corporations with ultra-liquid derivatives market depth and minimal tracking error.',
  QQQ: 'Concentrated exposure to secular technology innovators with high returns on invested capital and secular earnings compounding.',
  CPRX: 'Orphan drug commercial monopoly (FIRDAPSE) with high operating margins, pristine balance sheet, and long-dated patent exclusivity.',
  POWI: 'High-voltage Gallium Nitride (GaN) power conversion IC standard reducing power dissipation across hyperscale AI servers and EVs.',
  LNTH: 'Commercial diagnostic radiopharmaceutical monopoly with PYLARIFY PSMA-targeted PET imaging for precision oncology.',
  KO: 'World\'s leading nonalcoholic beverage bottling network with unmatched brand equity, global pricing power, and resilient dividend generation.',
  SBUX: 'Premier global specialty coffee retailer with 38,000+ stores, high-frequency digital rewards membership, and strong store-level unit economics.',
  JPM: 'Fortress balance sheet money-center financial institution dominating corporate investment banking, wealth management, and prime retail deposits.',
  V: 'Global payments duopoly operating high-margin tollbooth network processing trillions in consumer and commercial transactions.',
  DIS: 'Iconic intellectual property portfolio, premier global theme parks, and expanding direct-to-consumer streaming margins.',
  COST: 'Unrivaled warehouse club member loyalty with 92%+ renewal rates, negative working capital cycle, and massive bulk volume purchasing power.',
  WMT: 'World\'s largest omnichannel retailer combining massive physical store density with rapidly expanding high-margin digital advertising.',
  AMD: 'High-performance compute architectures challenging datacenter AI accelerators and server CPU market share.',
  ARM: 'Ubiquitous CPU instruction set architecture with expanding royalty rates per chip across mobile, auto, and hyperscale AI servers.',
  SMCI: 'Modular server architecture with engineering speed-to-market advantage in high-density liquid-cooled AI cluster deployments.',
  COIN: 'Leading US regulated digital asset gateway, institutional ETF custodian, and expanding Layer-2 blockchain transaction ecosystem.',
  VRT: 'Dominant pure-play provider of critical digital infrastructure, liquid cooling, and power management for AI data centers.',
  ISRG: 'Robotic-assisted minimally invasive surgical monopoly with high-margin recurring instrument and accessory revenue streams.',
  KLAC: 'Global monopoly in semiconductor process diagnostic inspection and metrology essential for advanced wafer fabrication yields.',
  CIEN: 'Niche monopoly in coherent optical networking vital for GPU cluster interconnects and data center scale-out.',
  ACLS: 'Specialized monopoly in high-energy ion implantation required for Silicon Carbide electric vehicle inverters.',
  TMDX: 'Disruptive warm perfusion technology revolutionizing heart, lung, and liver transplantation survival rates.',
  MEDP: 'High return-on-capital contract research organization catering exclusively to emerging biopharma.',
  ELF: 'Digitally-native fast-beauty disruptor taking rapid global market share with premium quality-to-price ratio.',
  DUOL: 'Gamified learning platform with organic user acquisition and accelerating ARPU conversion.',
};

export const CANONICAL_ASSET_RISKS: Record<string, string> = {
  NVDA: 'Hyperscaler capex digestion cycle, export control restrictions, and silicon supply bottlenecks.',
  AAPL: 'Global smartphone replacement cycle deceleration, regulatory app store fee headwinds, and China market competition.',
  MSFT: 'Enterprise IT budget compression, AI inference capex margin drag, and cloud consumption slowdown.',
  TSLA: 'Automotive gross margin compression from global EV price competition and autonomous regulatory hurdles.',
  PLTR: 'High multiple valuation sensitivity and government contract procurement timing lumpiness.',
  NVO: 'Compounding pharmacy copycat supply, Medicare price negotiations, and manufacturing fill-finish capacity bottlenecks.',
  LLY: 'Payer coverage restrictions, competitive GLP-1 pipeline readouts, and patent expiration timelines.',
  SPY: 'Macroeconomic recessionary downturns, multiple compression, and elevated market concentration in mega-cap tech.',
  QQQ: 'Elevated valuation multiples and interest rate discount rate sensitivity.',
  CPRX: 'Single-product revenue concentration risk and patent challenge litigation outcomes.',
  POWI: 'Consumer electronics cyclicality and silicon carbide replacement technology adoption.',
  LNTH: 'Single-source radioisotope reactor supply disruptions and generic diagnostic imaging competition.',
  KO: 'Foreign exchange currency translation headwinds and consumer discretionary spending shifts away from packaged goods.',
  SBUX: 'Unionization wage inflation, coffee commodity price fluctuations, and international consumer spending softness.',
  JPM: 'Credit loss provisions during recessionary credit cycles and commercial real estate exposure.',
  V: 'Credit Card Competition Act legislative pressures, interchange fee caps, and alternative payment networks.',
  DIS: 'Linear broadcast television cord-cutting acceleration and park attendance cyclicality.',
  COST: 'Consumer retail spending slowdown and international supply chain inventory disruptions.',
  WMT: 'Wage cost pressures, inventory shrink, and grocery price deflation headwinds.',
  AMD: 'Intense competitive pressure from NVIDIA in software ecosystem (CUDA) and Intel in client CPUs.',
  ARM: 'Customer in-house custom silicon architecture design risk (RISC-V) and licensing cycle timing.',
  SMCI: 'Supply chain component allocation delays and gross margin volatility in hyperscaler contracts.',
  COIN: 'Digital asset price volatility, regulatory staking classification scrutiny, and spot trading fee compression.',
  VRT: 'Supply chain electrical transformer delivery lead times and data center construction delays.',
  ISRG: 'Hospital capital expenditure freezes and competitive surgical robotic platform entrants.',
  KLAC: 'Semiconductor wafer fab equipment (WFE) spending cycle downturns and export restrictions.',
  CIEN: 'Telecom service provider capex reduction and component delivery lead times.',
  ACLS: 'Electric vehicle silicon carbide adoption slowdown and geographic fab buildout delays.',
  TMDX: 'Aviation logistics fuel price spikes and donor organ utilization variability.',
  MEDP: 'Early-stage biopharma venture funding contraction and clinical trial cancellation rates.',
  ELF: 'Color cosmetics fashion trend obsolescence and retail inventory de-stocking.',
  DUOL: 'Generative AI consumer competition and mobile app store subscription commission changes.',
};

export const CANONICAL_ASSET_CATALYSTS: Record<string, AssetCatalystProfile> = {
  NVDA: {
    trial: 'Blackwell GB200 NVL72 Rack-Scale Compute',
    phase: 'Mass Production & Hyperscale Delivery',
    timeline: 'Continuous FY26/27 Data Center Shipments',
    thesis: 'Dominant accelerated computing full-stack architecture with CUDA ecosystem standard and hardware moats.',
  },
  AAPL: {
    trial: 'Apple Intelligence On-Device AI Architecture',
    phase: 'Global iOS Rollout & Enterprise Services',
    timeline: 'Fall Product Cycle & Developer Conferences',
    thesis: 'Consumer hardware ecosystem with 2.2B active installed devices and high-margin recurring services growth.',
  },
  MSFT: {
    trial: 'Azure OpenAI Enterprise & Copilot Monetization',
    phase: 'Production Enterprise Scaling',
    timeline: 'Quarterly Cloud Consumption Reporting',
    thesis: 'Commercial enterprise software moat with mission-critical Azure infrastructure and security integrations.',
  },
  TSLA: {
    trial: 'Full Self-Driving (FSD) v12.5 & Robotaxi Network',
    phase: 'Commercial Validation & Autonomous Scaling',
    timeline: 'Cybercab Demonstration & Fleet Rollout',
    thesis: 'Next-gen electric vehicle platform and vision-only end-to-end neural network autonomy.',
  },
  PLTR: {
    trial: 'AIP (Artificial Intelligence Platform) Enterprise Bootcamps',
    phase: 'Commercial Acceleration',
    timeline: 'Q3/Q4 2026 Enterprise Expansion',
    thesis: 'Government defense and commercial ontology operating system enabling autonomous business workflows.',
  },
  NVO: {
    trial: 'CagriSema Phase 3 REDEFINE & Oral Amycretin',
    phase: 'Phase 3 Pivotal / Registration',
    timeline: 'Q4 2026 Phase 3 Trial Readouts',
    thesis: 'Secular obesity and diabetes therapeutic franchise with accelerating multi-billion manufacturing capacity.',
  },
  LLY: {
    trial: 'Zepbound Sleep Apnea Label & Orforglipron GLP-1',
    phase: 'Phase 3 Registration & FDA Filing',
    timeline: 'H2 2026 Regulatory Submissions',
    thesis: 'Global pharmaceutical market cap leader driving dual-incretin and triple-agonist metabolic platforms.',
  },
  KO: {
    trial: 'Global Volume Growth, Bottling System Refranchising & Direct-Store-Delivery',
    phase: 'Commercial Market Leadership & Margin Expansion',
    timeline: 'Quarterly Unit Volume & Pricing Power Readouts',
    thesis: 'World\'s preeminent beverage brand portfolio with unmatched global distribution bottling network and pricing power.',
  },
  SBUX: {
    trial: 'Triple Shot Reinvention, Store-Level Throughput & Digital Rewards Expansion',
    phase: 'Operational Turnaround & Unit Economics Acceleration',
    timeline: 'Quarterly Same-Store Sales (Comps) Reporting',
    thesis: 'Premier global specialty coffee brand driving customer throughput with 38,000+ locations and 34M+ active Rewards members.',
  },
  LNTH: {
    trial: 'PYLARIFY Imaging Volume & Alzheimer Diagnostic Pipeline',
    phase: 'Commercial / Expansion (Phase 3)',
    timeline: 'Q3 2026 Earnings & Product Roadmap',
    thesis: 'Market leader in diagnostic radiopharmaceuticals and PSMA-targeted PET imaging agents for prostate cancer.',
  },
  CPRX: {
    trial: 'FIRDAPSE Lambert-Eaton Myasthenic Syndrome Expansion',
    phase: 'Commercial Monopoly',
    timeline: 'FY26 Label Expansion Data',
    thesis: 'Rare neurological disease commercial franchise with pristine balance sheet and high cash conversion.',
  },
  POWI: {
    trial: 'GaN (Gallium Nitride) High-Voltage Power Conversion',
    phase: 'Data Center & Automotive Adoption',
    timeline: 'FY26 Server Efficiency Compliance Window',
    thesis: 'Energy-efficient power conversion ICs reducing phantom power loss across EVs, chargers, and servers.',
  },
  CIEN: {
    trial: 'WaveLogic 6 Nano 800G Coherent Optics',
    phase: 'Hyperscale AI Deployment',
    timeline: 'FY26 Optical Interconnect Ramp',
    thesis: 'Niche monopoly in coherent optical networking vital for GPU cluster interconnects and data center scale-out.',
  },
  ACLS: {
    trial: 'Purion Power SiC Ion Implantation Platform',
    phase: 'Automotive & Industrial Scaling',
    timeline: 'Q4 2026 Semiconductor Equipment Deliveries',
    thesis: 'Specialized monopoly in high-energy ion implantation required for Silicon Carbide electric vehicle inverters.',
  },
  TMDX: {
    trial: 'Organ Care System (OCS) Aviation Logistics Network',
    phase: 'National Clinical Standard of Care',
    timeline: 'Continuous OCS Flight Operations Expansion',
    thesis: 'Disruptive warm perfusion technology revolutionizing heart, lung, and liver transplantation survival rates.',
  },
  MEDP: {
    trial: 'Clinical Biotech Contract Research Acceleration',
    phase: 'Full-Service CRO Operations',
    timeline: 'Continuous RFP Backlog Delivery',
    thesis: 'High return-on-capital contract research organization catering exclusively to emerging biopharma.',
  },
  ELF: {
    trial: 'Global Retail Expansion & Skincare Integration',
    phase: 'International Rollout',
    timeline: 'UK/Europe Market Share Expansion',
    thesis: 'Digitally-native, fast-beauty disruptor taking rapid global market share with premium quality-to-price ratio.',
  },
  DUOL: {
    trial: 'Duolingo Max Generative AI Subscription Tiers',
    phase: 'Global Commercial Rollout',
    timeline: 'Continuous AI Course Launch',
    thesis: 'Gamified learning platform with organic user acquisition and accelerating ARPU conversion.',
  },
  JPM: {
    trial: 'Net Interest Margin (NIM) Optimization & Global Commercial Banking Expansion',
    phase: 'Tier-1 Money-Center Bank Scale',
    timeline: 'Quarterly Net Interest Income & Credit Provision Readouts',
    thesis: 'Fortress balance sheet money-center financial institution dominating corporate investment banking, wealth management, and prime retail deposits.',
  },
  V: {
    trial: 'Cross-Border Travel Volume & Real-Time Direct Settlement (Visa Direct)',
    phase: 'Global Digital Payment Processing Monopoly',
    timeline: 'Quarterly Payment Volume & Value-Added Services Growth',
    thesis: 'Global payments duopoly operating high-margin tollbooth network processing trillions in consumer and commercial transactions.',
  },
  DIS: {
    trial: 'Direct-to-Consumer (DTC) Streaming Profitability & Experiences Park Expansion',
    phase: 'DTC Margin Expansion & Cruise Fleet Scaling',
    timeline: 'Quarterly Disney+ Subscriber ARPU & Park Operating Income',
    thesis: 'Iconic intellectual property portfolio, premier global theme parks, and expanding direct-to-consumer streaming margins.',
  },
  COST: {
    trial: 'Membership Warehouse Global Expansion & Digital E-Commerce Fulfillment',
    phase: 'Global Club Format Scale',
    timeline: 'Quarterly Net Sales & Membership Fee Renewal Rates',
    thesis: 'Unrivaled warehouse club member loyalty with 92%+ renewal rates, negative working capital cycle, and massive bulk volume purchasing power.',
  },
  WMT: {
    trial: 'Walmart+ High-Margin Marketplace & Automated Supply Chain Fulfillment',
    phase: 'Omnichannel Retail Modernization',
    timeline: 'Quarterly Global E-Commerce & Retail Media (Walmart Connect) Ramp',
    thesis: 'World\'s largest omnichannel retailer combining massive physical store density with rapidly expanding high-margin digital advertising.',
  },
  AMD: {
    trial: 'Instinct MI350/MI400 AI Accelerator Rack Scaling & EPYC Server Dominance',
    phase: 'Hyperscale AI Accelerator Deliveries',
    timeline: 'Continuous FY26/27 Data Center Shipments',
    thesis: 'High-performance compute architectures challenging datacenter AI accelerators and server CPU market share.',
  },
  ARM: {
    trial: 'Armv9 Compute Subsystems (CSS) & Neoverse Enterprise Server Adoption',
    phase: 'Data Center & Automotive Licensing',
    timeline: 'FY26 Royalty Rate Escalation Window',
    thesis: 'Ubiquitous CPU instruction set architecture with expanding royalty rates per chip across mobile, auto, and hyperscale AI servers.',
  },
  SMCI: {
    trial: 'Direct Liquid Cooling (DLC) Hyperscale AI Server Cluster Integration',
    phase: 'High-Density Liquid Cooled Deployment',
    timeline: 'Continuous Rack-Scale Datacenter Shipments',
    thesis: 'Modular server architecture with engineering speed-to-market advantage in high-density liquid-cooled AI cluster deployments.',
  },
  COIN: {
    trial: 'Base L2 Layer-2 On-Chain Transaction Scaling & Institutional Custody Expansion',
    phase: 'Institutional Infrastructure Scaling',
    timeline: 'Continuous On-Chain Settlement Volume Readouts',
    thesis: 'Leading US regulated digital asset gateway, institutional ETF custodian, and expanding Layer-2 blockchain transaction ecosystem.',
  },
  VRT: {
    trial: 'Liquid Cooling & High-Density Datacenter Thermal Power Infrastructure',
    phase: 'Hyperscale Data Center Buildout',
    timeline: 'FY26 Power & Thermal Management Deliveries',
    thesis: 'Dominant pure-play provider of critical digital infrastructure, liquid cooling, and power management for AI data centers.',
  },
  ISRG: {
    trial: 'da Vinci 5 Next-Gen Robotic Surgical System Global Hospital Placements',
    phase: 'Commercial System Placement & Procedure Ramp',
    timeline: 'Quarterly Procedure Volume & Installed Base Growth',
    thesis: 'Robotic-assisted minimally invasive surgical monopoly with high-margin recurring instrument and accessory revenue streams.',
  },
  KLAC: {
    trial: 'Process Control & Optical Wafer Inspection for Advanced 2nm/GAA Nodes',
    phase: 'Sub-2nm Foundry Tool Shipments',
    timeline: 'Continuous Semiconductor Node Equipment Ramps',
    thesis: 'Global monopoly in semiconductor process diagnostic inspection and metrology essential for advanced wafer fabrication yields.',
  },
};

/** Get the authoritative, clean full corporate name */
export function getCanonicalAssetName(symbol: string, defaultName?: string): string {
  const upper = symbol.toUpperCase().replace("-USD", "").trim();
  return CANONICAL_ASSET_NAMES[upper] || defaultName || `${upper} Corporation`;
}

/** Get the authoritative fundamental moat thesis */
export function getCanonicalAssetMoat(symbol: string, defaultMoat?: string): string {
  const upper = symbol.toUpperCase().replace("-USD", "").trim();
  return (
    CANONICAL_ASSET_MOATS[upper] ||
    defaultMoat ||
    `${getCanonicalAssetName(upper)} demonstrates strong operational moats, disciplined capital allocation, and durable returns on invested capital.`
  );
}

/** Get the authoritative primary risk profile */
export function getCanonicalAssetRisk(symbol: string, defaultRisk?: string): string {
  const upper = symbol.toUpperCase().replace("-USD", "").trim();
  return (
    CANONICAL_ASSET_RISKS[upper] ||
    defaultRisk ||
    "Macroeconomic interest rate sensitivity, industry competitive shifts, and valuation multiple compression risk."
  );
}

function anyWordMatch(sec: string, ind: string, keywords: string[]): boolean {
  return keywords.some((kw) => sec.includes(kw) || ind.includes(kw));
}

/** Get domain-appropriate catalyst based on explicit registry or intelligent sector synthesis */
export function getCanonicalAssetCatalyst(
  symbol: string,
  sector: string = "",
  industry: string = "",
  companyName: string = ""
): AssetCatalystProfile {
  const upper = symbol.toUpperCase().replace("-USD", "").trim();
  if (CANONICAL_ASSET_CATALYSTS[upper]) {
    return CANONICAL_ASSET_CATALYSTS[upper];
  }

  const cleanName = companyName || getCanonicalAssetName(upper);
  const secLower = sector.toLowerCase();
  const indLower = industry.toLowerCase();

  // 1. Consumer Staples & Beverages
  if (anyWordMatch(secLower, indLower, ["beverage", "drink", "food", "tobacco", "staple", "consumer defensive"])) {
    return {
      trial: "Global Unit Volume, Retail Mix & Direct-Store-Delivery Execution",
      phase: "Commercial Market Leadership & Margin Expansion",
      timeline: "Quarterly Volume & Price/Mix Earnings Reports",
      thesis: `${cleanName} operates a resilient consumer distribution network with pricing power and dependable free cash flow conversion.`,
    };
  }

  // 2. Restaurants, Retail & Consumer Discretionary
  if (anyWordMatch(secLower, indLower, ["restaurant", "coffee", "retail", "consumer cyclical", "apparel", "luxury", "dining"])) {
    return {
      trial: "Same-Store Sales (Comps), Store-Level Throughput & Loyalty Growth",
      phase: "Unit Economics & Digital Membership Acceleration",
      timeline: "Quarterly Global Comparable Sales Readouts",
      thesis: `${cleanName} drives high recurring consumer transaction frequency, digital ordering growth, and disciplined unit expansion.`,
    };
  }

  // 3. Real Estate Investment Trusts (REITs) & Property
  if (anyWordMatch(secLower, indLower, ["reit", "real estate", "property", "lease", "mortgage", "housing"])) {
    return {
      trial: "Adjusted Funds From Operations (AFFO) Growth & Portfolio Occupancy",
      phase: "Capital Recycling & Net Lease Execution",
      timeline: "Quarterly AFFO Payout & Lease Renewal Reporting",
      thesis: `${cleanName} commands high-quality commercial real estate assets with long-term tenant leases and inflation-hedged dividend cash flows.`,
    };
  }

  // 4. Energy, Oil & Gas, Clean Power
  if (anyWordMatch(secLower, indLower, ["energy", "oil", "gas", "petroleum", "solar", "wind", "utility", "power", "pipeline"])) {
    return {
      trial: "Upstream Production Efficiency, LNG Export Expansion & Free Cash Flow Yield",
      phase: "Capital Discipline & Infrastructure Utilization",
      timeline: "Quarterly Barrel Equivalents & Dividend/Buyback Updates",
      thesis: `${cleanName} benefits from disciplined capital allocation, low-cost extraction assets, and resilient commodity cash conversion.`,
    };
  }

  // 5. Materials, Mining & Metals
  if (anyWordMatch(secLower, indLower, ["material", "mining", "gold", "copper", "steel", "chemical", "metal", "lithium"])) {
    return {
      trial: "All-In Sustaining Cost (AISC) Margin Optimization & Mineral Reserve Life",
      phase: "Tier-1 Mine Production & Smelting Operations",
      timeline: "Quarterly Ore Grade & Ton Yield Reporting",
      thesis: `${cleanName} operates tier-1 low-cost extraction assets with multi-decade reserve life and strong commodity cycle leverage.`,
    };
  }

  // 6. Financial Services & Banking
  if (anyWordMatch(secLower, indLower, ["financial", "bank", "credit", "insurance", "broker", "asset management", "capital markets"])) {
    return {
      trial: "Net Interest Margin (NIM) Expansion & Fee Asset Under Management Growth",
      phase: "Capital Management & Prime Lending Scale",
      timeline: "Quarterly Net Interest Income & Credit Loss Readouts",
      thesis: `${cleanName} demonstrates fortress capital adequacy, diversified institutional fee revenue, and prudent credit underwriting.`,
    };
  }

  // 7. Healthcare & Biotechnology
  if (anyWordMatch(secLower, indLower, ["health", "biotech", "pharma", "clinical", "therapeutic", "medical", "hospital"])) {
    return {
      trial: "Clinical Pipeline Registrations & FDA Regulatory Approval Cycle",
      phase: "Pivotal Clinical Trial & Commercial Launch",
      timeline: "Quarterly Regulatory Submissions & Trial Endpoints",
      thesis: `${cleanName} advances high-unmet-need medical therapies protected by strong intellectual property moats.`,
    };
  }

  // 8. Default Commercial Product & TAM Expansion
  return {
    trial: "Commercial Execution, TAM Expansion & Operating Margin Compounding",
    phase: "Market Scaling & Product Line Optimization",
    timeline: "Quarterly Earnings & Capital Allocation Guidance",
    thesis: `${cleanName} demonstrates solid balance sheet quality, strong operational execution, and consistent institutional accumulation.`,
  };
}
