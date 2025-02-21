import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_simulation_plot(t, y, labels, title):
    """Create interactive plot for simulation results"""
    fig = make_subplots(rows=len(y[0]), cols=1, 
                       subplot_titles=[f"{label} over Time" for label in labels])
    
    for i in range(len(y[0])):
        fig.add_trace(
            go.Scatter(x=t, y=y[:,i], name=labels[i]),
            row=i+1, col=1
        )
    
    fig.update_layout(
        height=200*len(y[0]),
        title_text=title,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def export_results(t, y, labels, filename):
    """Export simulation results to CSV"""
    data = {'Time': t}
    for i, label in enumerate(labels):
        data[label] = y[:,i]
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df
