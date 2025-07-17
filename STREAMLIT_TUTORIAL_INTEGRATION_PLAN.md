# Enhanced Streamlit Dashboard for Tutorial Completion
*Implementation Plan for In-App Tutorial Experience*

## 🎯 **Phase 1: Tutorial Mode Integration**

### **1. Add Tutorial Sidebar Section**
```python
# dashboard/app.py additions
TUTORIAL_NAMES = {
    1: "Implement a Scenario",
    2: "Interpret the Metrics", 
    3: "Visualise the Results",
    4: "Export Charts",
    5: "Generate Custom Visualisations",
    6: "Implement a New Agent",
    7: "Customise Visual Style",
    8: "Stress-Test Your Assumptions",
    9: "Save Everything with Export Bundles",
    10: "Explore the Chart Gallery"
}

def tutorial_mode():
    st.sidebar.markdown("## 🎓 Tutorial Mode")
    tutorial_enabled = st.sidebar.checkbox("Enable Tutorial Mode")
    
    if tutorial_enabled:
        tutorial_num = st.sidebar.selectbox(
            "Select Tutorial", 
            range(1, 11),
            format_func=lambda x: f"Tutorial {x}: {TUTORIAL_NAMES[x]}"
        )
        
        # Tutorial progress tracking
        progress_key = f"tutorial_{tutorial_num}_progress"
        if progress_key not in st.session_state:
            st.session_state[progress_key] = 0
            
        return tutorial_num
    return None
```

### **2. Tutorial 1: In-App Scenario Implementation**
```python
def tutorial_1_implement_scenario():
    st.markdown("# Tutorial 1: Implement a Scenario")
    
    # Step 1: Configuration Selection
    st.markdown("## Step 1: Choose Configuration")
    config_source = st.radio(
        "Configuration Source",
        ["Use Template", "Upload File", "Edit Parameters"]
    )
    
    if config_source == "Use Template":
        template = st.selectbox(
            "Select Template",
            ["params_template.yml", "parameters_template.csv"]
        )
        config_data = load_template(template)
        
    elif config_source == "Edit Parameters":
        config_data = parameter_editor()
        
    # Step 2: Index Data
    st.markdown("## Step 2: Provide Index Data")
    index_file = st.file_uploader(
        "Upload Index CSV", 
        type=['csv'],
        help="CSV with Date column and Monthly_TR or Return column"
    )
    
    # Step 3: Run Simulation
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            results = run_simulation(config_data, index_file)
            st.session_state.simulation_results = results
            st.success("Simulation complete!")
            
    # Tutorial completion tracking
    if 'simulation_results' in st.session_state:
        st.session_state.tutorial_1_progress = 100
        st.balloons()
```

### **3. Tutorial 2: Metrics Interpretation with Guidance**
```python
def tutorial_2_interpret_metrics():
    st.markdown("# Tutorial 2: Interpret the Metrics")
    
    if 'simulation_results' not in st.session_state:
        st.warning("Complete Tutorial 1 first to generate results")
        return
        
    results = st.session_state.simulation_results
    
    # Interactive metrics exploration
    st.markdown("## Key Metrics Analysis")
    
    # Risk/Return Analysis
    with st.expander("📊 Risk/Return Trade-offs"):
        st.markdown("""
        **What to look for:**
        - Higher `AnnReturn` with manageable `AnnVol` 
        - Compare efficiency across agents
        """)
        
        # Highlight best performers
        best_return = results['AnnReturn'].idxmax()
        best_sharpe = (results['AnnReturn'] / results['AnnVol']).idxmax()
        
        st.info(f"**Best Return**: {best_return} ({results.loc[best_return, 'AnnReturn']:.2%})")
        st.info(f"**Best Risk-Adjusted**: {best_sharpe}")
    
    # Threshold Analysis with Traffic Lights
    with st.expander("🚦 Threshold Analysis"):
        te_threshold = 0.03  # 3% TE cap
        shortfall_threshold = 0.05  # 5% shortfall limit
        
        for agent in results.index:
            te = results.loc[agent, 'TrackingErr']
            shortfall = results.loc[agent, 'ShortfallProb']
            
            # Traffic light colors
            te_color = "🔴" if te > te_threshold else "🟢"
            shortfall_color = "🔴" if shortfall > shortfall_threshold else "🟢"
            
            st.markdown(f"""
            **{agent}**:
            - Tracking Error: {te_color} {te:.2%} (limit: {te_threshold:.1%})
            - Shortfall Prob: {shortfall_color} {shortfall:.2%} (limit: {shortfall_threshold:.1%})
            """)
```

### **4. Tutorial 3: Enhanced Visualization with Export**
```python
def tutorial_3_visualise_results():
    st.markdown("# Tutorial 3: Visualise the Results")
    
    # Standard dashboard charts with tutorial guidance
    chart_type = st.selectbox(
        "Select Chart Type",
        ["Risk-Return Scatter", "Funding Fan", "Path Distribution", "Correlation Heatmap"]
    )
    
    # Chart-specific guidance
    if chart_type == "Risk-Return Scatter":
        st.markdown("""
        **How to read this chart:**
        - X-axis: Tracking Error (lower is better)
        - Y-axis: Annual Return (higher is better) 
        - Color: Shortfall Probability (green = low risk)
        - Sweet spot: High return, low tracking error, low shortfall risk
        """)
        
    # Render chart with enhanced interactivity
    fig = create_chart(chart_type, st.session_state.simulation_results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    st.markdown("## Export This Chart")
    export_formats = st.multiselect(
        "Export Formats", 
        ["PNG", "HTML", "PDF", "PPTX"],
        default=["HTML"]
    )
    
    alt_text = st.text_input("Alt Text", "Risk-return analysis chart")
    
    if st.button("Export Chart"):
        export_chart(fig, chart_type, export_formats, alt_text)
        st.success("Chart exported successfully!")
```

## 🎯 **Phase 2: Parameter Sweep Integration**

### **5. Tutorial Mode for Parameter Sweeps**
```python
def parameter_sweep_tutorial():
    st.markdown("# Parameter Sweep Mode")
    
    # Mode selection with descriptions
    sweep_mode = st.selectbox(
        "Sweep Mode",
        ["capital", "returns", "alpha_shares", "vol_mult"],
        format_func=lambda x: f"{x.title()}: {SWEEP_DESCRIPTIONS[x]}"
    )
    
    # Dynamic parameter configuration
    if sweep_mode == "capital":
        st.markdown("## Capital Allocation Optimization")
        ext_pa_min = st.number_input("External PA Min (MM)", value=100)
        ext_pa_max = st.number_input("External PA Max (MM)", value=500)
        ext_pa_step = st.number_input("External PA Step (MM)", value=50)
        
        # Similar for other parameters
        
    # Preview sweep scenarios
    scenarios = generate_sweep_scenarios(sweep_mode, parameters)
    st.markdown(f"**{len(scenarios)} scenarios will be generated**")
    
    with st.expander("Preview Scenarios"):
        st.dataframe(scenarios.head(10))
        
    # Run sweep
    if st.button("Run Parameter Sweep"):
        with st.spinner(f"Running {len(scenarios)} scenarios..."):
            progress_bar = st.progress(0)
            results = run_parameter_sweep_with_progress(scenarios, progress_bar)
            st.session_state.sweep_results = results
            st.success("Parameter sweep complete!")
```

## 🔧 **Implementation Files Needed**

### **1. Enhanced Dashboard Structure**
```
dashboard/
├── app.py (enhanced with tutorial mode)
├── tutorials/
│   ├── __init__.py
│   ├── tutorial_1.py (scenario implementation)
│   ├── tutorial_2.py (metrics interpretation)
│   ├── tutorial_3.py (visualization)
│   ├── tutorial_4.py (export charts)
│   ├── tutorial_5.py (custom visualizations)
│   └── parameter_sweeps.py (sweep tutorials)
├── components/
│   ├── config_editor.py
│   ├── export_manager.py
│   └── progress_tracker.py
```

### **2. Tutorial State Management**
```python
# dashboard/state_manager.py
class TutorialStateManager:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        if 'tutorial_progress' not in st.session_state:
            st.session_state.tutorial_progress = {i: 0 for i in range(1, 11)}
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = None
        if 'sweep_results' not in st.session_state:
            st.session_state.sweep_results = None
    
    def mark_tutorial_complete(self, tutorial_num: int):
        st.session_state.tutorial_progress[tutorial_num] = 100
        
    def get_progress(self, tutorial_num: int) -> int:
        return st.session_state.tutorial_progress.get(tutorial_num, 0)
```

## 🚀 **Key Benefits of This Approach**

1. **🎯 Guided Learning**: Step-by-step tutorials with interactive feedback
2. **🔄 No CLI Required**: Everything works within familiar web interface  
3. **📊 Real-time Results**: Immediate visualization of changes and exports
4. **🎓 Progress Tracking**: Users can see completion status across tutorials
5. **🔧 Parameter Sweep Integration**: Advanced tutorials work seamlessly with multi-scenario analysis
6. **📱 Accessible**: Works on any device with web browser, no terminal needed
7. **🎨 Export Integration**: Charts can be exported directly from the tutorial interface

This approach transforms the tutorials from CLI-based exercises into an interactive, guided learning experience that's much more accessible to typical users while maintaining all the powerful functionality of the parameter sweep engine.
