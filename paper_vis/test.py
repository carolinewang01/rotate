import plotly.graph_objects as go

r = [1, 5, 2, 2, 1]
theta = [0, 90, 180, 270, 360]

fig = go.Figure(data=go.Scatterpolar(r=r, theta=theta, mode='lines'))

pdf_path = "results/neurips_figures/test.pdf"
fig.write_image(pdf_path, scale=2)