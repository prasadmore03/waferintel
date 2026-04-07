"""
AI-Powered Wafer Defect Detection + RAG Assistant
RAG-based AI assistant for wafer defect explanation using rule-based system
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass

@dataclass
class DefectExplanation:
    """Structured output for defect explanation"""
    defect_name: str
    description: str
    root_causes: List[str]
    process_step: str
    equipment_causes: List[str]
    yield_impact: str
    prevention_methods: List[str]
    inspection_methods: List[str]
    severity_level: str
    confidence_level: str
    recommended_action: str

class WaferDefectKnowledgeBase:
    """Knowledge base for wafer defect information"""
    
    def __init__(self):
        self.defect_database = self._create_defect_database()
        self.process_steps = self._create_process_steps_database()
        self.equipment_issues = self._create_equipment_database()
        
    def _create_defect_database(self) -> Dict[str, Dict]:
        """Create comprehensive defect database"""
        return {
            "Center": {
                "description": "Defects concentrated in the center region of the wafer, typically circular or oval patterns affecting the central area.",
                "root_causes": [
                    "Temperature gradients across wafer during processing",
                    "Non-uniform chemical deposition in center",
                    "Contamination buildup in central processing zones",
                    "Equipment alignment issues",
                    "Spin coating non-uniformity"
                ],
                "process_step": "Photolithography, Chemical Vapor Deposition (CVD), Spin Coating",
                "equipment_causes": [
                    "Hot plate temperature non-uniformity",
                    "Spin coater misalignment",
                    "Photolithography stepper centering issues",
                    "CVD reactor flow distribution problems"
                ],
                "yield_impact": "High - Center defects often affect critical device areas",
                "prevention_methods": [
                    "Optimize temperature uniformity",
                    "Improve spin coating parameters",
                    "Regular equipment calibration",
                    "Enhance cleaning procedures",
                    "Process parameter optimization"
                ],
                "inspection_methods": [
                    "Automated optical inspection (AOI)",
                    "Scanning electron microscopy (SEM)",
                    "Wafer mapping analysis",
                    "In-line metrology"
                ],
                "severity": "High",
                "recommended_action": "Immediate equipment check and process parameter review"
            },
            
            "Donut": {
                "description": "Ring-shaped defects avoiding the center, forming a donut pattern around the wafer periphery.",
                "root_causes": [
                    "Edge-to-center temperature gradients",
                    "Chemical depletion during processing",
                    "Spin coating edge effects",
                    "Etch rate non-uniformity",
                    "Gas flow distribution issues"
                ],
                "process_step": "Etching, Deposition, Spin Coating, Cleaning",
                "equipment_causes": [
                    "Etch chamber gas flow imbalance",
                    "Spin coater acceleration profile issues",
                    "Temperature hot spots at edges",
                    "Chemical delivery system problems"
                ],
                "yield_impact": "Medium to High - Depends on ring width and location",
                "prevention_methods": [
                    "Optimize gas flow dynamics",
                    "Adjust spin coating parameters",
                    "Improve temperature uniformity",
                    "Modify chemical delivery timing",
                    "Edge bead removal optimization"
                ],
                "inspection_methods": [
                    "Ring pattern analysis",
                    "Radial defect mapping",
                    "Cross-sectional SEM",
                    "Process control monitoring"
                ],
                "severity": "Medium to High",
                "recommended_action": "Process parameter optimization and equipment maintenance"
            },
            
            "Edge-Loc": {
                "description": "Localized defects at specific edge locations, often appearing as spots or small clusters near wafer perimeter.",
                "root_causes": [
                    "Edge contamination during handling",
                    "Localized stress concentrations",
                    "Edge bead formation irregularities",
                    "Chuck mark damage",
                    "Edge coating non-uniformity"
                ],
                "process_step": "Wafer Handling, Edge Bead Removal, Photolithography",
                "equipment_causes": [
                    "Robot arm misalignment",
                    "Chuck pressure irregularities",
                    "Edge bead removal tool issues",
                    "Carrier wafer contamination",
                    "Loading mechanism problems"
                ],
                "yield_impact": "Medium - Often affects peripheral devices",
                "prevention_methods": [
                    "Improve handling procedures",
                    "Regular chuck maintenance",
                    "Optimize edge bead removal",
                    "Enhanced cleaning protocols",
                    "Carrier wafer quality control"
                ],
                "inspection_methods": [
                    "Edge-specific AOI",
                    "High-resolution edge scanning",
                    "Contact inspection",
                    "Process monitoring"
                ],
                "severity": "Medium",
                "recommended_action": "Handling procedure review and equipment maintenance"
            },
            
            "Edge-Ring": {
                "description": "Continuous ring defects along the wafer edge, forming a complete or partial ring around the perimeter.",
                "root_causes": [
                    "Systematic edge processing issues",
                    "Temperature edge effects",
                    "Chemical edge accumulation",
                    "Mechanical edge stress",
                    "Process edge bead problems"
                ],
                "process_step": "Edge Trimming, Cleaning, Deposition, Etching",
                "equipment_causes": [
                    "Edge trim tool wear",
                    "Temperature gradient at edges",
                    "Chemical edge pooling",
                    "Chuck edge damage",
                    "Process tool edge alignment"
                ],
                "yield_impact": "High - Affects entire edge region",
                "prevention_methods": [
                    "Regular edge tool maintenance",
                    "Temperature profile optimization",
                    "Chemical flow adjustment",
                    "Edge process parameter tuning",
                    "Enhanced edge cleaning"
                ],
                "inspection_methods": [
                    "Edge ring pattern analysis",
                    "Continuous edge monitoring",
                    "Ring width measurement",
                    "Process control charts"
                ],
                "severity": "High",
                "recommended_action": "Immediate edge process review and equipment maintenance"
            },
            
            "Loc": {
                "description": "Localized spot defects at specific locations, appearing as isolated points or small clusters.",
                "root_causes": [
                    "Particle contamination",
                    "Localized equipment malfunction",
                    "Chemical splatter",
                    "Mask defects",
                    "Process parameter spikes"
                ],
                "process_step": "Photolithography, Deposition, Etching, Cleaning",
                "equipment_causes": [
                    "Filter failure in chemical systems",
                    "Mask alignment errors",
                    "Nozzle clogging",
                    "Sensor malfunctions",
                    "Tool contamination"
                ],
                "yield_impact": "Variable - Depends on location and size",
                "prevention_methods": [
                    "Enhanced filtration systems",
                    "Regular mask inspection",
                    "Nozzle maintenance schedule",
                    "Sensor calibration",
                    "Clean room protocol enforcement"
                ],
                "inspection_methods": [
                    "Point defect mapping",
                    "High-magnification inspection",
                    "Statistical process control",
                    "Root cause analysis"
                ],
                "severity": "Low to High",
                "recommended_action": "Identify root cause and implement targeted prevention"
            },
            
            "Near-full": {
                "description": "Large area defects covering most of the wafer surface, indicating widespread process issues.",
                "root_causes": [
                    "Systemic process failures",
                    "Batch contamination",
                    "Equipment-wide malfunction",
                    "Chemical bath degradation",
                    "Environmental contamination"
                ],
                "process_step": "Multiple process steps affected",
                "equipment_causes": [
                    "Chemical bath exhaustion",
                    "HVAC system failures",
                    "Temperature control system failure",
                    "Batch processing equipment issues",
                    "Clean room contamination"
                ],
                "yield_impact": "Very High - Affects majority of wafer",
                "prevention_methods": [
                    "Regular chemical bath maintenance",
                    "Environmental monitoring",
                    "Equipment preventive maintenance",
                    "Batch quality control",
                    "Clean room integrity checks"
                ],
                "inspection_methods": [
                    "Full wafer scanning",
                    "Yield analysis",
                    "Process parameter monitoring",
                    "Environmental sampling"
                ],
                "severity": "Critical",
                "recommended_action": "Immediate process shutdown and comprehensive investigation"
            },
            
            "Random": {
                "description": "Scattered defects distributed randomly across the wafer surface with no clear pattern.",
                "root_causes": [
                    "Random particle contamination",
                    "Statistical process variations",
                    "Equipment random failures",
                    "Environmental fluctuations",
                    "Material quality variations"
                ],
                "process_step": "Multiple process steps",
                "equipment_causes": [
                    "Filter degradation",
                    "Random equipment malfunctions",
                    "Environmental control fluctuations",
                    "Material batch variations",
                    "Tool wear patterns"
                ],
                "yield_impact": "Variable - Depends on defect density",
                "prevention_methods": [
                    "Statistical process control",
                    "Enhanced filtration",
                    "Environmental monitoring",
                    "Material quality control",
                    "Predictive maintenance"
                ],
                "inspection_methods": [
                    "Statistical defect analysis",
                    "Pattern recognition algorithms",
                    "Process control charts",
                    "Yield trend analysis"
                ],
                "severity": "Variable",
                "recommended_action": "Statistical analysis and process optimization"
            },
            
            "Scratch": {
                "description": "Linear defects caused by mechanical contact, appearing as lines or scratches on wafer surface.",
                "root_causes": [
                    "Mechanical handling damage",
                    "Tool contact during processing",
                    "Particle-induced scratching",
                    "Robot arm misalignment",
                    "Wafer carrier issues"
                ],
                "process_step": "Wafer Handling, Loading/Unloading, Transport",
                "equipment_causes": [
                    "Robot arm calibration issues",
                    "Chuck surface damage",
                    "Carrier wafer contamination",
                    "Handling tool wear",
                    "Transport system misalignment"
                ],
                "yield_impact": "High - Often creates critical device failures",
                "prevention_methods": [
                    "Handling procedure optimization",
                    "Regular equipment maintenance",
                    "Chuck surface inspection",
                    "Carrier wafer quality control",
                    "Robot arm calibration"
                ],
                "inspection_methods": [
                    "Scratch detection algorithms",
                    "Surface profilometry",
                    "Microscopic inspection",
                    "Contact inspection"
                ],
                "severity": "High",
                "recommended_action": "Immediate handling procedure review and equipment maintenance"
            },
            
            "None": {
                "description": "No detectable defects - wafer appears normal and defect-free.",
                "root_causes": [
                    "Normal processing conditions",
                    "Optimal process parameters",
                    "Good equipment condition",
                    "Clean environment",
                    "Quality materials"
                ],
                "process_step": "All process steps within specifications",
                "equipment_causes": [
                    "Equipment functioning properly",
                    "No mechanical issues",
                    "Optimal process conditions",
                    "Regular maintenance effective"
                ],
                "yield_impact": "Positive - High yield expected",
                "prevention_methods": [
                    "Maintain current process parameters",
                    "Continue preventive maintenance",
                    "Monitor process stability",
                    "Quality control procedures"
                ],
                "inspection_methods": [
                    "Standard inspection protocols",
                    "Quality assurance checks",
                    "Process monitoring",
                    "Yield verification"
                ],
                "severity": "None",
                "recommended_action": "Continue current processing and monitoring"
            }
        }
    
    def _create_process_steps_database(self) -> Dict[str, List[str]]:
        """Create process steps database"""
        return {
            "Photolithography": [
                "Photoresist coating",
                "Soft bake",
                "Exposure",
                "Post-exposure bake",
                "Development",
                "Hard bake"
            ],
            "Etching": [
                "Etch preparation",
                "Pattern transfer",
                "Etch monitoring",
                "End point detection",
                "Cleaning"
            ],
            "Deposition": [
                "Surface preparation",
                "Material deposition",
                "Thickness control",
                "Uniformity check",
                "Post-deposition treatment"
            ],
            "Cleaning": [
                "Pre-clean",
                "Chemical cleaning",
                "Rinse cycles",
                "Drying",
                "Inspection"
            ]
        }
    
    def _create_equipment_database(self) -> Dict[str, List[str]]:
        """Create equipment issues database"""
        return {
            "Hot Plate": [
                "Temperature non-uniformity",
                "Calibration drift",
                "Surface contamination",
                "Hot spots"
            ],
            "Spin Coater": [
                "Misalignment",
                "Acceleration issues",
                "Vibration",
                "Contamination"
            ],
            "Stepper": [
                "Alignment errors",
                "Focus issues",
                "Exposure problems",
                "Mask defects"
            ],
            "Etch Chamber": [
                "Gas flow imbalance",
                "Pressure variations",
                "Temperature issues",
                "Contamination"
            ],
            "CVD Reactor": [
                "Flow distribution problems",
                "Temperature gradients",
                "Pressure control",
                "Chemical delivery"
            ]
        }
    
    def get_defect_info(self, defect_name: str) -> Optional[Dict]:
        """Get defect information from database"""
        return self.defect_database.get(defect_name)
    
    def search_similar_defects(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar defects based on query"""
        defect_names = list(self.defect_database.keys())
        defect_descriptions = [self.defect_database[name]["description"] for name in defect_names]
        
        # Simple keyword matching for now
        query_lower = query.lower()
        scores = []
        
        for name, description in zip(defect_names, defect_descriptions):
            score = 0
            if query_lower in name.lower():
                score += 10
            if any(word in description.lower() for word in query_lower.split()):
                score += 5
            scores.append((name, score))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class WaferRAGAssistant:
    """RAG-based AI Assistant for wafer defect explanation"""
    
    def __init__(self):
        """Initialize RAG Assistant using rule-based approach"""
        self.knowledge_base = WaferDefectKnowledgeBase()
        print("✅ Rule-based RAG Assistant initialized successfully")
    
    def generate_explanation(self, predicted_defect: str, dataset_source: str, 
                           confidence: float = 0.0) -> DefectExplanation:
        """
        Generate detailed defect explanation using RAG approach
        
        Args:
            predicted_defect: Predicted defect type
            dataset_source: Source dataset (WM811K or Mixed-type)
            confidence: Model confidence score
            
        Returns:
            Structured defect explanation
        """
        
        # Get defect information from knowledge base
        defect_info = self.knowledge_base.get_defect_info(predicted_defect)
        
        if defect_info is None:
            # Fallback for unknown defects
            return DefectExplanation(
                defect_name=predicted_defect,
                description="Unknown defect type - insufficient information available",
                root_causes=["Unknown", "Insufficient data"],
                process_step="Unknown",
                equipment_causes=["Unknown", "Insufficient data"],
                yield_impact="Unknown",
                prevention_methods=["Further investigation needed"],
                inspection_methods=["Additional analysis required"],
                severity_level="Unknown",
                confidence_level=f"{confidence:.1%}",
                recommended_action="Consult with process engineering team"
            )
        
        # Determine confidence level
        if confidence >= 0.8:
            confidence_level = "High"
        elif confidence >= 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        # Create structured explanation
        explanation = DefectExplanation(
            defect_name=predicted_defect,
            description=defect_info["description"],
            root_causes=defect_info["root_causes"],
            process_step=defect_info["process_step"],
            equipment_causes=defect_info["equipment_causes"],
            yield_impact=defect_info["yield_impact"],
            prevention_methods=defect_info["prevention_methods"],
            inspection_methods=defect_info["inspection_methods"],
            severity_level=defect_info["severity"],
            confidence_level=confidence_level,
            recommended_action=defect_info["recommended_action"]
        )
        
        return explanation
    
    def format_explanation(self, explanation: DefectExplanation) -> str:
        """Format explanation in structured output format"""
        formatted_output = f"""
🔍 **DEFECT ANALYSIS REPORT**

**Defect Name:** {explanation.defect_name}
**Dataset Source:** {explanation.defect_name}
**Confidence Level:** {explanation.confidence_level}

---

📋 **DESCRIPTION**
{explanation.description}

---

🔍 **ROOT CAUSES**
{chr(10).join(f"• {cause}" for cause in explanation.root_causes)}

---

⚙️ **PROCESS STEP**
{explanation.process_step}

---

🔧 **EQUIPMENT CAUSES**
{chr(10).join(f"• {cause}" for cause in explanation.equipment_causes)}

---

📉 **YIELD IMPACT**
{explanation.yield_impact}

---

🛡️ **PREVENTION METHODS**
{chr(10).join(f"• {method}" for method in explanation.prevention_methods)}

---

🔬 **INSPECTION METHODS**
{chr(10).join(f"• {method}" for method in explanation.inspection_methods)}

---

⚠️ **SEVERITY LEVEL**
{explanation.severity_level}

---

🎯 **RECOMMENDED ACTION**
{explanation.recommended_action}

---

*Generated by AI-Powered Wafer Defect Detection + RAG Assistant*
"""
        return formatted_output
    
    def chat_with_assistant(self, user_query: str, context: Optional[Dict] = None) -> str:
        """
        Chat interface for assistant
        
        Args:
            user_query: User's question
            context: Optional context from CNN predictions
            
        Returns:
            Assistant response
        """
        
        # Extract defect information from context if available
        if context and 'defect_type' in context:
            predicted_defect = context['defect_type']
            confidence = context.get('confidence', 0.0)
            dataset_source = context.get('dataset', 'Unknown')
            
            # Generate explanation
            explanation = self.generate_explanation(predicted_defect, dataset_source, confidence)
            return self.format_explanation(explanation)
        
        # Handle general queries with enhanced pattern matching
        query_lower = user_query.lower()
        
        # Check for specific defect types
        defect_keywords = {
            'center': 'Center',
            'donut': 'Donut', 
            'edge': 'Edge',
            'loc': 'Loc',
            'near-full': 'Near-full',
            'random': 'Random',
            'scratch': 'Scratch'
        }
        
        for keyword, defect_name in defect_keywords.items():
            if keyword in query_lower:
                defect_info = self.knowledge_base.get_defect_info(defect_name)
                if defect_info:
                    return self._format_defect_response(defect_name, defect_info)
        
        # Check for process-related questions
        if any(word in query_lower for word in ['process', 'manufacturing', 'production', 'fabrication']):
            return self._format_process_response(query_lower)
        
        # Check for equipment-related questions
        if any(word in query_lower for word in ['equipment', 'machine', 'tool', 'device']):
            return self._format_equipment_response(query_lower)
        
        # Check for prevention-related questions
        if any(word in query_lower for word in ['prevent', 'avoid', 'stop', 'reduce']):
            return self._format_prevention_response(query_lower)
        
        # Check for inspection-related questions
        if any(word in query_lower for word in ['inspect', 'check', 'detect', 'find']):
            return self._format_inspection_response(query_lower)
        
        # Check for yield-related questions
        if any(word in query_lower for word in ['yield', 'production', 'output', 'efficiency']):
            return self._format_yield_response(query_lower)
        
        # Check for severity-related questions
        if any(word in query_lower for word in ['severity', 'critical', 'impact', 'serious']):
            return self._format_severity_response(query_lower)
        
        # General semiconductor manufacturing response
        return self._format_general_response(query_lower)
    
    def _format_defect_response(self, defect_name: str, defect_info: Dict) -> str:
        """Format defect-specific response"""
        return f"""
 **{defect_name} Defect Analysis**

**Description:**
{defect_info['description']}

**Common Root Causes:**
{chr(10).join(f"• {cause}" for cause in defect_info['root_causes'])}

**Affected Process Steps:**
{defect_info['process_step']}

**Equipment-Related Issues:**
{chr(10).join(f"• {issue}" for issue in defect_info['equipment_causes'])}

**Impact on Yield:**
{defect_info['yield_impact']}

**Prevention Methods:**
{chr(10).join(f"• {method}" for method in defect_info['prevention_methods'])}

**Inspection Techniques:**
{chr(10).join(f"• {method}" for method in defect_info['inspection_methods'])}

**Severity Level:** {defect_info['severity']}

**Recommended Action:** {defect_info['recommended_action']}

---

*Would you like more specific information about any aspect of this defect?*
"""
    
    def _format_process_response(self, query: str) -> str:
        """Format process-related response"""
        return """
 **Semiconductor Process Information**

**Key Manufacturing Steps:**

1. **Photolithography**
   - Photoresist coating and exposure
   - Pattern transfer to wafer
   - Common defects: Center, Edge-Loc

2. **Chemical Vapor Deposition (CVD)**
   - Thin film deposition
   - Temperature and gas flow control
   - Common defects: Donut, Center

3. **Etching**
   - Pattern etching and material removal
   - Chemical and plasma etching
   - Common defects: Edge-Ring, Random

4. **Cleaning**
   - Wafer surface preparation
   - Chemical cleaning processes
   - Common defects: Particle contamination

5. **Inspection**
   - Automated optical inspection
   - Metrology and measurements
   - Quality control checks

**Process Control Tips:**
- Monitor temperature uniformity across all steps
- Maintain chemical bath purity and concentration
- Regular equipment calibration and maintenance
- Clean room environment control (ISO Class 5-7)

*Ask about specific process steps for detailed information!*
"""
    
    def _format_equipment_response(self, query: str) -> str:
        """Format equipment-related response"""
        return """
 **Semiconductor Equipment Information**

**Critical Equipment Systems:**

1. **Photolithography Tools**
   - Steppers/Scanners
   - Mask aligners
   - Track systems
   - Hot plates for baking

2. **Deposition Systems**
   - CVD reactors (thermal, plasma, PECVD)
   - Sputtering systems
   - Evaporation tools
   - ALD (Atomic Layer Deposition)

3. **Etching Equipment**
   - Wet etch benches
   - Dry etch (RIE) systems
   - Plasma etchers
   - End-point detection systems

4. **Inspection Tools**
   - Automated Optical Inspection (AOI)
   - Scanning Electron Microscopes (SEM)
   - Metrology tools (profilometry, ellipsometry)

**Equipment Maintenance Best Practices:**
- Daily calibration checks
- Preventive maintenance schedules
- Regular cleaning protocols
- Performance monitoring and trend analysis
- Spare parts inventory management

**Common Equipment-Related Defects:**
- Temperature non-uniformity → Center, Donut defects
- Gas flow issues → Edge-Ring, Random defects
- Mechanical misalignment → Edge-Loc, Scratch defects
- Contamination buildup → Loc, Near-full defects

*Ask about specific equipment types for detailed troubleshooting!*
"""
    
    def _format_prevention_response(self, query: str) -> str:
        """Format prevention-related response"""
        return """
 **Defect Prevention Strategies**

**Prevention Categories:**

1. **Environmental Control**
   - Clean room protocols (ISO Class 5-7)
   - Temperature and humidity control
   - Air filtration and particle monitoring
   - Personnel training and protocols

2. **Process Optimization**
   - Parameter tuning and control
   - Statistical Process Control (SPC)
   - Recipe optimization
   - Real-time monitoring systems

3. **Equipment Maintenance**
   - Regular calibration schedules
   - Preventive maintenance programs
   - Component replacement schedules
   - Performance trend analysis

4. **Material Quality Control**
   - Chemical purity verification
   - Gas quality monitoring
   - Substrate quality checks
   - Supplier qualification programs

5. **Handling Procedures**
   - Automated handling systems
   - Proper wafer carriers
   - Robot arm calibration
   - Contact minimization protocols

**Specific Prevention Methods by Defect Type:**

- **Center Defects**: Optimize temperature uniformity, improve spin coating
- **Donut Defects**: Adjust gas flow dynamics, modify process timing
- **Edge Defects**: Improve handling procedures, maintain chuck surfaces
- **Scratch Defects**: Enhance handling protocols, regular equipment inspection
- **Random Defects**: Improve filtration, environmental monitoring

**Prevention Metrics:**
- Defect density reduction targets
- Yield improvement goals
- Process capability indices (Cpk, Cpk)
- Cost of quality measurements

*Ask about specific defect types for targeted prevention strategies!*
"""
    
    def _format_inspection_response(self, query: str) -> str:
        """Format inspection-related response"""
        return """
 **Wafer Inspection Techniques**

**Inspection Methods:**

1. **Automated Optical Inspection (AOI)**
   - High-speed image capture and analysis
   - Pattern recognition algorithms
   - Defect classification and mapping
   - Real-time process monitoring

2. **Scanning Electron Microscopy (SEM)**
   - High-magnification defect analysis
   - Sub-micron defect detection
   - Cross-sectional analysis
   - Root cause investigation

3. **Metrology and Surface Analysis**
   - Profilometry for surface topography
   - Ellipsometry for film thickness
   - Atomic Force Microscopy (AFM)
   - Surface roughness measurement

4. **Electrical Testing**
   - Probe card testing
   - Wafer-level electrical parameters
   - Device-level functionality testing
   - Yield analysis

**Inspection Strategy:**

**In-line Inspection:**
- Real-time monitoring during production
- Immediate feedback for process control
- Early defect detection and correction
- Statistical process control integration

**Off-line Inspection:**
- Detailed sample analysis
- Root cause investigation
- Process qualification and validation
- New process development support

**Defect Classification:**
- Pattern recognition algorithms
- Machine learning classification
- Expert system rules
- Human-in-the-loop verification

**Inspection Best Practices:**
- Regular calibration of inspection tools
- Proper lighting and optics setup
- Algorithm optimization and validation
- Operator training and skill development

*Ask about specific inspection techniques or defect types!*
"""
    
    def _format_yield_response(self, query: str) -> str:
        """Format yield-related response"""
        return """
 **Yield Management and Analysis**

**Yield Metrics:**

1. **Wafer Yield**
   - Good die / Total die
   - Functional yield percentage
   - Parametric yield analysis
   - Yield loss categorization

2. **Defect Density**
   - Defects per wafer
   - Defects per unit area
   - Critical defect mapping
   - Trend analysis over time

3. **Process Capability**
   - Cpk (Process Capability Index)
   - Process spread analysis
   - Specification compliance
   - Statistical control limits

**Yield Improvement Strategies:**

**Systematic Approach:**
1. **Data Collection**
   - Comprehensive defect tracking
   - Process parameter monitoring
   - Equipment performance data
   - Environmental conditions logging

2. **Analysis and Prioritization**
   - Pareto analysis of defect types
   - Root cause identification
   - Impact assessment on yield
   - Cost-benefit analysis

3. **Implementation and Monitoring**
   - Process parameter optimization
   - Equipment upgrades or modifications
   - Procedure changes and training
   - Continuous improvement cycles

**Yield Targets by Industry:**
- **Mature Processes**: 90-95% yield
- **Leading Edge**: 70-85% yield
- **Memory Devices**: 85-92% yield
- **Logic Devices**: 80-90% yield

**Yield Loss Categories:**
- Pattern related defects (40-50% of losses)
- Particle contamination (20-30% of losses)
- Equipment related (15-25% of losses)
- Process parameter issues (10-15% of losses)

*Ask about specific yield improvement strategies!*
"""
    
    def _format_severity_response(self, query: str) -> str:
        """Format severity-related response"""
        return """
 **Defect Severity Assessment**

**Severity Classification System:**

** Critical (Immediate Action Required)**
- **Near-full defects**: Affect most of wafer surface
- **Systemic failures**: Batch-wide contamination
- **Equipment breakdown**: Major process failures
- **Yield Impact**: Very High (>50% loss)

** High (Priority Action Required)**
- **Center defects**: Affect critical device areas
- **Edge-Ring defects**: Complete edge region affected
- **Scratch defects**: Create critical device failures
- **Yield Impact**: High (20-50% loss)

** Medium (Scheduled Action)**
- **Donut defects**: Depends on ring width and location
- **Edge-Loc defects**: Affect peripheral devices
- **Random defects**: Depends on defect density
- **Yield Impact**: Medium (5-20% loss)

** Low (Monitor and Plan)**
- **Loc defects**: Depends on location and size
- **Isolated defects**: Minimal impact
- **Minor variations**: Within specification limits
- **Yield Impact**: Low (<5% loss)

**Severity Assessment Criteria:**

**Impact Factors:**
1. **Device Criticality**: Location relative to active areas
2. **Defect Size**: Physical dimensions and area affected
3. **Electrical Impact**: Effect on device functionality
4. **Yield Loss**: Percentage of good die lost
5. **Production Impact**: Line stoppage or slowdown

**Response Time Guidelines:**
- **Critical**: Immediate (minutes to 1 hour)
- **High**: Urgent (1-4 hours)
- **Medium**: Scheduled (4-24 hours)
- **Low**: Planned (24-72 hours)

**Decision Matrix:**
- **Critical + High Yield Loss**: Stop production, immediate investigation
- **High + Medium Yield Loss**: Continue with monitoring, quick investigation
- **Medium + Low Yield Loss**: Document, schedule investigation
- **Low + Minimal Impact**: Monitor, trend analysis

*Ask about specific severity scenarios or response protocols!*
"""
    
    def _format_general_response(self, query: str) -> str:
        """Format general semiconductor response with better keyword matching"""
        
        # Enhanced keyword matching for broader coverage
        query_lower = query.lower()
        
        # Check for specific semiconductor terms
        semiconductor_terms = {
            'wafer': """
🔍 **Wafer Manufacturing Information**

**Wafer Basics:**
- Silicon wafers are the foundation of semiconductor devices
- Typical sizes: 100mm, 150mm, 200mm, 300mm diameters
- Thickness: 525-775μm for standard wafers
- Crystal orientation: <100>, <111> for different applications

**Wafer Processing Steps:**
1. **Wafer Cleaning**: Remove contaminants
2. **Oxidation**: Grow silicon dioxide layer
3. **Photolithography**: Pattern transfer
4. **Etching**: Remove unwanted material
5. **Deposition**: Add new material layers
6. **Doping**: Introduce impurities
7. **Metallization**: Add interconnects
8. **Packaging**: Final device protection

**Common Wafer Defects:**
- Center defects, Donut patterns, Edge defects
- Scratches, Random defects, Near-full coverage
- Each defect type has specific causes and prevention methods

*Ask me about specific wafer defects or processing steps!*
""",
            'silicon': """
⚗️ **Silicon Semiconductor Information**

**Silicon Properties:**
- Crystal structure: Diamond cubic
- Band gap: 1.12 eV (room temperature)
- Melting point: 1414°C
- Electrical properties: Can be doped to create p-type or n-type

**Silicon Wafer Production:**
1. **Crystal Growth**: Czochralski (CZ) or Float Zone (FZ)
2. **Wafer Slicing**: Diamond wire sawing
3. **Lapping**: Remove saw damage
4. **Polishing**: Create mirror surface
5. **Cleaning**: Remove contaminants

**Silicon Processing:**
- **Oxidation**: Si + O₂ → SiO₂ (thermal oxide)
- **Doping**: Add boron (p-type) or phosphorus (n-type)
- **Diffusion**: High temperature dopant introduction
- **Ion Implantation**: Precise dopant placement

**Silicon Defect Types:**
- Crystal defects: Dislocations, stacking faults
- Process defects: Pattern defects, contamination
- Mechanical defects: Scratches, cracks

*Ask about specific silicon processes or defect types!*
""",
            'chip': """
🔧 **Chip/Device Manufacturing Information**

**Chip Fabrication Process:**
1. **Wafer Preparation**: Clean and prepare silicon wafer
2. **Front-End Processing**: Create transistors and circuits
3. **Back-End Processing**: Add interconnects and metallization
4. **Testing**: Electrical and functional testing
5. **Packaging**: Protect and connect the chip

**Front-End Steps:**
- **Well formation**: Create p-wells and n-wells
- **Gate formation**: Create transistor gates
- **Source/Drain**: Create transistor terminals
- **Isolation**: Separate active regions

**Back-End Steps:**
- **Dielectric deposition**: Add insulating layers
- **Metallization**: Add metal interconnects
- **Passivation**: Protect final device
- **Bond pads**: Create connection points

**Common Chip Defects:**
- Pattern defects: Missing or extra features
- Electrical defects: Shorts, opens, leakage
- Mechanical defects: Cracks, delamination

*Ask about specific chip fabrication steps or defect types!*
""",
            'fabrication': """
🏭 **Semiconductor Fabrication Information**

**Fabrication Facility (Fab) Requirements:**
- **Clean Room**: ISO Class 5-7 environment
- **Temperature Control**: 20-25°C ± 0.5°C
- **Humidity Control**: 30-50% RH
- **Air Filtration**: HEPA/ULPA filtration systems

**Key Fabrication Areas:**
1. **Lithography**: Pattern creation and transfer
2. **Etch**: Material removal and pattern definition
3. **Deposition**: Thin film growth and coating
4. **Diffusion/Implant**: Dopant introduction
5. **Metrology**: Measurement and inspection
6. **Test**: Electrical and functional testing

**Process Control:**
- **Statistical Process Control (SPC)**: Monitor process variations
- **Recipe Management**: Standardized process parameters
- **Equipment Monitoring**: Real-time performance tracking
- **Yield Management**: Track and improve production yield

**Common Fabrication Issues:**
- Particle contamination
- Equipment drift
- Process parameter variations
- Material quality issues

*Ask about specific fabrication processes or quality control methods!*
""",
            'quality': """
📊 **Quality Control in Semiconductor Manufacturing**

**Quality Control Methods:**

1. **In-line Quality Control**
   - Real-time process monitoring
   - Automated inspection systems
   - Statistical process control charts
   - Immediate feedback and correction

2. **Off-line Quality Control**
   - Sample analysis and testing
   - Detailed defect investigation
   - Process qualification studies
   - New process validation

**Quality Metrics:**
- **Defect Density**: Defects per cm²
- **Yield**: Good die / Total die
- **Process Capability (Cpk)**: Process variation vs. specification
- **Mean Time Between Failures (MTBF)**: Equipment reliability

**Quality Tools:**
- **Pareto Analysis**: Identify major defect types
- **Control Charts**: Monitor process stability
- **Design of Experiments (DOE)**: Optimize processes
- **Failure Mode Analysis**: Prevent defects

**Quality Standards:**
- **ISO 9001**: Quality management systems
- **IATF 16949**: Automotive quality standards
- **JEDEC Standards**: Industry specifications

*Ask about specific quality control techniques or metrics!*
""",
            'control': """
⚙️ **Process Control in Semiconductor Manufacturing**

**Process Control Elements:**

1. **Parameter Control**
   - Temperature: ±1°C precision
   - Pressure: ±0.1 Torr accuracy
   - Flow rates: ±1% accuracy
   - Time: ±1 second precision

2. **Equipment Control**
   - Calibration schedules
   - Maintenance procedures
   - Performance monitoring
   - Automated recipe execution

3. **Environmental Control**
   - Clean room classification
   - Temperature/humidity control
   - Vibration isolation
   - Electrostatic discharge (ESD) protection

**Control Systems:**
- **SPC (Statistical Process Control)**: Monitor process variation
- **APC (Advanced Process Control)**: Real-time adjustment
- **FDC (Fault Detection and Classification)**: Identify issues
- **EES (Equipment Engineering System)**: Equipment management

**Control Metrics:**
- **Process Capability Index (Cpk)**: Process vs. specification
- **Process Stability**: Control chart analysis
- **Equipment Utilization**: OEE (Overall Equipment Effectiveness)
- **First Pass Yield**: Success rate without rework

**Control Best Practices:**
- Regular calibration and maintenance
- Standard operating procedures (SOPs)
- Training and certification programs
- Continuous improvement cycles

*Ask about specific control methods or process parameters!*
"""
        }
        
        # Check for specific terms
        for term, response in semiconductor_terms.items():
            if term in query_lower:
                return response
        
        # Check for general help/introduction requests
        if any(word in query_lower for word in ['help', 'hello', 'hi', 'introduction', 'about', 'what can you do']):
            return """
🤖 **AI Semiconductor Manufacturing Assistant**

Hello! I'm your specialized AI assistant for semiconductor manufacturing and wafer defect analysis. I can help you with:

**🔍 Defect Analysis:**
- Detailed explanations of all defect types
- Root cause analysis and prevention strategies
- Equipment-related issues and solutions
- Process optimization recommendations

**⚙️ Process Information:**
- Photolithography, etching, deposition processes
- Equipment specifications and maintenance
- Quality control procedures and standards
- Yield improvement strategies

**🔬 Inspection & Testing:**
- Automated optical inspection methods
- Microscopy and metrology techniques
- Defect classification algorithms
- Statistical process control

**📊 Technical Support:**
- Process troubleshooting
- Equipment maintenance procedures
- Quality control methodologies
- Industry best practices

**How to Use:**
1. **Ask specific questions**: "What causes center defects?"
2. **Request explanations**: "Explain donut defect in detail"
3. **Seek advice**: "How can I prevent edge defects?"
4. **Get help**: "What equipment causes scratches?"

**Context-Aware Feature:**
I also provide detailed analysis based on your wafer defect predictions from the CNN models. Upload wafer images for AI-powered explanations!

*Feel free to ask any question about semiconductor manufacturing - I'm here to help!*
"""
        
        # Check for troubleshooting requests
        if any(word in query_lower for word in ['troubleshoot', 'problem', 'issue', 'error', 'fix', 'solve']):
            return """
🔧 **Semiconductor Troubleshooting Guide**

**Common Troubleshooting Areas:**

1. **Defect Pattern Analysis**
   - Identify defect type and location
   - Check process parameter correlations
   - Review equipment status and maintenance
   - Analyze environmental conditions

2. **Process Issues**
   - Parameter drift or variation
   - Chemical bath degradation
   - Temperature control problems
   - Equipment performance issues

3. **Equipment Problems**
   - Mechanical misalignment
   - Calibration drift
   - Contamination buildup
   - Component wear or failure

**Troubleshooting Methodology:**
1. **Data Collection**: Gather process and defect data
2. **Pattern Analysis**: Identify trends and correlations
3. **Root Cause Analysis**: Use 5-Why or Fishbone diagrams
4. **Solution Implementation**: Apply corrective actions
5. **Verification**: Confirm problem resolution

**Quick Fixes:**
- High defect density → Check filtration and cleaning
- Pattern defects → Verify photolithography alignment
- Temperature issues → Check hot plate uniformity
- Random defects → Review clean room protocols

*Describe your specific problem for targeted troubleshooting advice!*
"""
        
        # Default comprehensive response
        return """
🧠 **AI Semiconductor Manufacturing Assistant**

I can help you with comprehensive semiconductor manufacturing knowledge! Here are some topics I can assist with:

**🔍 Defect Analysis:**
Ask about any of the 8 defect types:
- "What causes center defects?"
- "Explain scratch defects"
- "How to prevent edge defects?"

**⚙️ Process Information:**
- "How does photolithography work?"
- "Explain CVD process"
- "What's in etching process?"

**🔧 Equipment & Troubleshooting:**
- "What equipment causes defects?"
- "How to maintain steppers?"
- "Troubleshoot temperature issues"

**📊 Quality & Yield:**
- "How to improve yield?"
- "Quality control methods"
- "Process capability analysis"

**🔬 Inspection & Testing:**
- "Best inspection techniques"
- "How to detect defects?"
- "Metrology methods"

**💬 Try asking:**
- Specific defect types (center, donut, scratch, etc.)
- Process steps (photolithography, etching, deposition)
- Equipment issues (temperature, alignment, contamination)
- Quality topics (yield, control, inspection)

**Context-Aware Feature:**
Upload wafer images for AI-powered defect analysis with detailed explanations based on CNN predictions!

*What specific semiconductor manufacturing topic would you like to know about?*
"""

# Example usage and testing
if __name__ == "__main__":
    # Initialize the RAG assistant
    assistant = WaferRAGAssistant()
    # Test with example defect
    test_defect = "Donut"
    test_dataset = "Mixed-type"
    test_confidence = 0.85
    
    explanation = assistant.generate_explanation(test_defect, test_dataset, test_confidence)
    print(assistant.format_explanation(explanation))
    
    # Test chat interface
    context = {
        'defect_type': 'Center',
        'confidence': 0.92,
        'dataset': 'WM811K'
    }
    
    response = assistant.chat_with_assistant("Explain this defect", context)
    print(response)
