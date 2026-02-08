/**
 * D3.js Visualizations for Orbital Survey
 */

// Color Distribution Bar Chart
function createColorChart(colors, containerId) {
    const container = d3.select(containerId);
    container.selectAll("*").remove();

    const width = container.node().getBoundingClientRect().width;
    const height = Math.max(400, colors.length * 50);
    const margin = { top: 10, right: 120, bottom: 10, left: 10 };

    const svg = container.append("svg")
        .attr("width", width)
        .attr("height", height);

    const maxPct = d3.max(colors, d => d.percentage);
    const barWidth = width - margin.left - margin.right - 120;

    colors.forEach((color, i) => {
        const y = i * (height / colors.length);
        const barHeight = (height / colors.length) - 10;

        const g = svg.append("g")
            .attr("class", "color-bar")
            .attr("transform", `translate(${margin.left}, ${y})`);

        // Background bar
        g.append("rect")
            .attr("class", "color-bar-bg")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", barWidth)
            .attr("height", barHeight)
            .attr("rx", 6);

        // Colored bar (animated)
        const fillWidth = (color.percentage / maxPct) * barWidth;
        g.append("rect")
            .attr("class", "color-bar-fill")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", 0)
            .attr("height", barHeight)
            .attr("rx", 6)
            .attr("fill", color.hex)
            .attr("stroke", d3.color(color.hex).darker(0.5))
            .attr("stroke-width", 2)
            .transition()
            .duration(1000)
            .delay(i * 100)
            .attr("width", fillWidth);

        // Percentage text
        g.append("text")
            .attr("class", "color-label")
            .attr("x", barWidth + 15)
            .attr("y", barHeight / 2)
            .attr("dy", "0.35em")
            .attr("fill", "#00b4a0")
            .text(`${color.percentage.toFixed(1)}%`);

        // Hex code
        g.append("text")
            .attr("class", "color-hex")
            .attr("x", barWidth + 15)
            .attr("y", barHeight / 2 + 16)
            .attr("dy", "0.35em")
            .text(color.hex);

        // Hover effect
        g.on("mouseover", function() {
            d3.select(this).select(".color-bar-fill")
                .transition()
                .duration(200)
                .attr("opacity", 0.8);
        })
        .on("mouseout", function() {
            d3.select(this).select(".color-bar-fill")
                .transition()
                .duration(200)
                .attr("opacity", 1);
        });
    });
}

// Hue-Saturation Scatter Plot
function createScatterPlot(scatterData, dominantColors, containerId, tooltipId) {
    const container = d3.select(containerId);
    container.selectAll("*").remove();

    const width = container.node().getBoundingClientRect().width;
    const height = 500;
    const margin = { top: 30, right: 30, bottom: 80, left: 60 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const svg = container.append("svg")
        .attr("width", width)
        .attr("height", height);

    const g = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
        .domain([0, 360])
        .range([0, plotWidth]);

    const yScale = d3.scaleLinear()
        .domain([0, 100])
        .range([plotHeight, 0]);

    // Background
    g.append("rect")
        .attr("width", plotWidth)
        .attr("height", plotHeight)
        .attr("fill", "white")
        .attr("stroke", "#ddd");

    // Grid lines
    const gridLines = g.append("g").attr("class", "grid");

    // Vertical grid
    for (let i = 0; i <= 10; i++) {
        const x = (i / 10) * plotWidth;
        gridLines.append("line")
            .attr("x1", x)
            .attr("y1", 0)
            .attr("x2", x)
            .attr("y2", plotHeight)
            .attr("stroke", "#f0f0f0")
            .attr("stroke-width", 1);
    }

    // Horizontal grid
    for (let i = 0; i <= 5; i++) {
        const y = (i / 5) * plotHeight;
        gridLines.append("line")
            .attr("x1", 0)
            .attr("y1", y)
            .attr("x2", plotWidth)
            .attr("y2", y)
            .attr("stroke", "#f0f0f0")
            .attr("stroke-width", 1);
    }

    // Plot scatter points
    const tooltip = d3.select(tooltipId);

    g.selectAll(".scatter-dot")
        .data(scatterData)
        .enter()
        .append("circle")
        .attr("class", "scatter-dot")
        .attr("cx", d => xScale(d.hue))
        .attr("cy", d => yScale(d.saturation))
        .attr("r", 0)
        .attr("fill", d => `rgb(${d.rgb[0]}, ${d.rgb[1]}, ${d.rgb[2]})`)
        .attr("opacity", d => 0.3 + d.value / 200)
        .on("mouseover", function(event, d) {
            const hex = `#${d.rgb[0].toString(16).padStart(2, '0')}${d.rgb[1].toString(16).padStart(2, '0')}${d.rgb[2].toString(16).padStart(2, '0')}`;
            tooltip.html(`
                <div>
                    <span class="tooltip-color" style="background: ${hex}"></span>
                    <strong>${hex.toUpperCase()}</strong>
                </div>
                <div style="margin-top: 8px;">
                    H: ${d.hue.toFixed(0)}°<br>
                    S: ${d.saturation.toFixed(0)}%<br>
                    V: ${d.value.toFixed(0)}%
                </div>
            `)
            .style("left", (event.pageX + 15) + "px")
            .style("top", (event.pageY - 15) + "px")
            .classed("visible", true);
        })
        .on("mouseout", function() {
            tooltip.classed("visible", false);
        })
        .transition()
        .duration(800)
        .delay((d, i) => i * 0.5)
        .attr("r", 2.5);

    // Plot dominant colors as larger circles
    dominantColors.forEach(color => {
        g.append("circle")
            .attr("class", "scatter-dominant")
            .attr("cx", xScale(color.hue))
            .attr("cy", yScale(color.saturation))
            .attr("r", 0)
            .attr("fill", color.hex)
            .transition()
            .duration(800)
            .attr("r", 8);
    });

    // Hue rainbow bar
    const rainbowHeight = 16;
    const rainbowY = plotHeight + 15;

    const rainbowGradient = svg.append("defs")
        .append("linearGradient")
        .attr("id", "hue-gradient")
        .attr("x1", "0%")
        .attr("x2", "100%");

    for (let i = 0; i <= 10; i++) {
        const hue = i / 10;
        const [r, g, b] = hsvToRgb(hue, 0.9, 0.95);
        rainbowGradient.append("stop")
            .attr("offset", `${i * 10}%`)
            .attr("stop-color", `rgb(${r}, ${g}, ${b})`);
    }

    g.append("rect")
        .attr("x", 0)
        .attr("y", rainbowY)
        .attr("width", plotWidth)
        .attr("height", rainbowHeight)
        .attr("fill", "url(#hue-gradient)")
        .attr("stroke", "#aaa");

    // Axes
    const xAxis = d3.axisBottom(xScale)
        .tickValues([0, 90, 180, 270, 360]);
    const yAxis = d3.axisLeft(yScale)
        .tickValues([0, 20, 40, 60, 80, 100])
        .tickFormat(d => d + "%");

    g.append("g")
        .attr("class", "axis")
        .attr("transform", `translate(0, ${plotHeight})`)
        .call(xAxis);

    g.append("g")
        .attr("class", "axis")
        .call(yAxis);

    // Axis labels
    g.append("text")
        .attr("class", "axis-label")
        .attr("x", plotWidth / 2)
        .attr("y", rainbowY + rainbowHeight + 35)
        .attr("text-anchor", "middle")
        .text("HUE (degrees)");

    g.append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("x", -plotHeight / 2)
        .attr("y", -45)
        .attr("text-anchor", "middle")
        .text("SATURATION");
}

// Helper: HSV to RGB
function hsvToRgb(h, s, v) {
    let r, g, b;
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);

    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }

    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}
