# Draft Agents.md Tutorial Section Update - Implementation Dependent

## � **CRITICAL DEPENDENCY NOTICE**
**This tutorial restructuring plan is dependent on parameter sweep engine implementation.** The current CLI only supports single-scenario runs. Tutorial updates should be implemented AFTER parameter sweep functionality is added to the codebase.

**Implementation Sequence**:
1. **FIRST**: Parameter sweep engine implementation (4 modes: capital, returns, alpha_shares, vol_mult)
2. **THEN**: Tutorial restructuring based on working parameter sweep capabilities  
3. **FINALLY**: Tutorial testing and validation with actual sweep functionality

## �📋 **TUTORIAL RESTRUCTURING REQUIREMENTS** (Post-Implementation)

Based on user testing of Tutorial 1, the following issues were identified and need to be addressed AFTER parameter sweep implementation:

### **Current Tutorial 1 Problems:** (Identified but dependent on parameter sweep implementation)
1. **Missing Mode Parameter Documentation** - Tutorial doesn't explain the `--mode` parameter (will be functional post-implementation)
2. **Template Selection Confusion** - Multiple templates with no guidance on which to choose
3. **Parameter Sweep Integration Gap** - Tutorial shows old single-scenario approach, needs to introduce sweep functionality (post-implementation)
4. **User Experience Flow** - Jumps directly to complex examples without building understanding

### **REQUIRED: Complete Tutorial 1 Restructure**

**New Tutorial 1 Structure (5 Parts):**

#### **Part 1: Basic Program Operation**
- **Objective**: Run simplest version, establish baseline understanding
- **Content**:
  - Single scenario run using `params_template.yml`
  - Basic CLI command without mode parameter
  - Console output interpretation
  - Excel file structure explanation
  - Common troubleshooting issues
- **Success Criteria**: User successfully runs program and understands basic metrics

#### **Part 2: Capital Mode Introduction**  
- **Objective**: Introduce parameter sweep concept with capital allocation
- **Content**:
  - Explain `--mode=capital` parameter
  - Use `capital_mode_template.csv` 
  - Show how multiple scenarios are generated
  - Compare single vs. sweep results
  - Excel output differences (multiple sheets vs. single)
- **Success Criteria**: User understands sweep concept and capital allocation variations

#### **Part 3: Returns Mode Introduction**
- **Objective**: Explore return/volatility assumption variations
- **Content**:
  - Explain `--mode=returns` parameter
  - Use `returns_mode_template.csv`
  - Demonstrate return/volatility sensitivity analysis
  - Risk-return chart interpretation
  - When to use returns mode vs. capital mode
- **Success Criteria**: User can set up and interpret returns sensitivity analysis

#### **Part 4: Alpha Shares Mode Introduction**
- **Objective**: Understand alpha/beta split optimization
- **Content**:
  - Explain `--mode=alpha_shares` parameter  
  - Use `alpha_shares_mode_template.csv`
  - Alpha vs. beta share allocation concepts
  - Performance optimization through share splits
  - Tracking error implications
- **Success Criteria**: User understands alpha/beta optimization concepts

#### **Part 5: Vol Mult Mode Introduction**
- **Objective**: Stress testing through volatility scaling
- **Content**:
  - Explain `--mode=vol_mult` parameter
  - Use `vol_mult_mode_template.csv`
  - Stress testing methodology
  - Risk scenario analysis
  - Integration with other modes
- **Success Criteria**: User can perform comprehensive stress testing

### **Additional Requirements:**

#### **Template Documentation Enhancement**
Each template CSV needs clear documentation:
- **Purpose**: When to use this mode
- **Parameters**: What each column controls
- **Expected Output**: What to look for in results
- **Next Steps**: How to interpret and act on results

#### **User Experience Improvements**
- **Progressive Complexity**: Each part builds on previous knowledge
- **Clear Examples**: Working commands for each mode
- **Troubleshooting**: Common errors and solutions
- **Integration Guide**: How modes work together

#### **Console Output Enhancement**
- **Mode Identification**: CLI should clearly state which mode is running
- **Progress Indicators**: Show sweep progress for multi-scenario runs
- **Result Summary**: Clear distinction between single vs. sweep outputs

---

## 📝 **IMPLEMENTATION INSTRUCTIONS FOR CODEX**

### **Primary Task**: Restructure Tutorial 1 in docs/UserGuide.md

**Location**: `/workspaces/Portable-Alpha-Extension-Model/docs/UserGuide.md`
**Section**: "Introductory Tutorial 1 – Implement a Scenario" (around line 91)

#### **Step 1: Replace Existing Tutorial 1**
- **Current Content**: Single tutorial with basic CLI command
- **New Content**: 5-part tutorial structure as outlined above
- **Format**: Maintain existing markdown structure and formatting

#### **Step 2: Create Template Documentation**
- **Location**: Add new section before Tutorial 1
- **Content**: Comprehensive guide to each CSV template
- **Format**: Table or structured list with purpose, parameters, examples

#### **Step 3: Update CLI Examples**
- **Current**: Shows old-style single scenario commands
- **New**: Progressive examples showing each mode
- **Requirements**: Working commands that users can copy-paste

#### **Step 4: Console Output Examples**
- **Add**: Sample console output for each mode
- **Format**: Code blocks showing expected CLI responses
- **Purpose**: Set user expectations for each tutorial part

#### **Step 5: Cross-Reference Updates**
- **Tutorial 2**: Update to reference 5-part Tutorial 1 structure
- **Tutorial 3**: Ensure visualization examples work with all modes
- **Navigation**: Update tutorial roadmap to reflect new structure

### **Quality Standards**
- **Clarity**: Each part should be understandable by complete beginners
- **Completeness**: Working examples for every command
- **Consistency**: Uniform formatting and terminology
- **Testing**: All commands should be verified to work
- **User-Focused**: Written from learner perspective, not developer perspective

### **Success Criteria**
1. **Part 1 Completion**: User can run basic simulation and understand output
2. **Mode Understanding**: User grasps parameter sweep concept and mode differences  
3. **Template Familiarity**: User knows which template to use for different scenarios
4. **Integration Ready**: User prepared for Tutorials 2-3 with any mode
5. **Self-Sufficient**: User can troubleshoot common issues independently

---

## 🎯 **PRIORITY CLASSIFICATION**

**Priority**: 🔥 **CRITICAL** - Blocks new user onboarding
**Effort**: 📏 **MEDIUM** - Requires content restructuring, not code changes
**Impact**: 🎯 **HIGH** - Improves entire tutorial experience
**Dependencies**: ✅ **NONE** - Parameter sweep engine already complete

This update addresses the fundamental user experience gap identified during tutorial testing and ensures new users can successfully learn and use the system.
