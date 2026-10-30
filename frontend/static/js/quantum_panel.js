/**
 * Quantum Panel Module - Autonomous Smart Traffic Management
 * Visualizes Qiskit QAOA quantum optimization telemetry:
 * 5-qubit circuit metadata, optimal basis state bitstring decoder,
 * Chart.js probability distribution, 5x5 QUBO coupling matrix heatmap,
 * and manual QAOA solver triggers.
 */

(function () {
    'use strict';

    class QuantumPanel {
        constructor() {
            this.chart = null;
            this.heatmapCanvas = document.getElementById('quboHeatmapCanvas');
            this.lastBitstring = '';
        }

        init() {
            this.initChart();
            this.initHeatmap();
            this.bindEvents();
            console.log('[QuantumPanel] Initialized.');
        }

        bindEvents() {
            const btnSolve = document.getElementById('btnQuantumSolve');
            if (btnSolve) {
                btnSolve.addEventListener('click', () => this.triggerSolve());
            }
        }

        triggerSolve() {
            const btn = document.getElementById('btnQuantumSolve');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Executing QAOA Hamiltonian...';
            }

            fetch('/api/quantum_solve', { method: 'POST' })
                .then(res => res.json())
                .then(data => {
                    console.log('[QuantumPanel] QAOA Solve response:', data);
                    setTimeout(() => {
                        if (btn) {
                            btn.disabled = false;
                            btn.innerHTML = '<i class="fa-solid fa-microchip"></i> Run Manual QAOA Solver';
                        }
                    }, 500);
                })
                .catch(err => {
                    console.error('[QuantumPanel] Error in QAOA solve:', err);
                    if (btn) {
                        btn.disabled = false;
                        btn.innerHTML = '<i class="fa-solid fa-microchip"></i> Run Manual QAOA Solver';
                    }
                });
        }

        /**
         * Initialize Chart.js for basis state probabilities
         */
        initChart() {
            const ctx = document.getElementById('qaoaChart');
            if (!ctx || typeof Chart === 'undefined') return;

            const defaultStates = ['|01010>', '|01000>', '|11010>', '|00010>', '|01011>', '|10101>', '|00000>', '|11111>'];
            const defaultProbs = [0.42, 0.18, 0.12, 0.08, 0.07, 0.06, 0.04, 0.03];

            this.chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: defaultStates,
                    datasets: [{
                        label: 'State Probability',
                        data: defaultProbs,
                        backgroundColor: defaultStates.map((s, idx) => idx === 0 ? 'rgba(0, 255, 136, 0.85)' : 'rgba(0, 212, 255, 0.45)'),
                        borderColor: defaultStates.map((s, idx) => idx === 0 ? '#00ff88' : '#00d4ff'),
                        borderWidth: 1,
                        borderRadius: 3
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => `Probability: ${(ctx.parsed.y * 100).toFixed(1)}%`
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#94a3b8', font: { family: 'Consolas', size: 9 } },
                            grid: { display: false }
                        },
                        y: {
                            beginAtZero: true,
                            max: 1.0,
                            ticks: { color: '#64748b', font: { size: 9 }, stepSize: 0.25 },
                            grid: { color: '#1e293b' }
                        }
                    }
                }
            });
        }

        /**
         * Initialize and draw the 5x5 QUBO coupling heatmap
         */
        initHeatmap() {
            this.drawHeatmap(null);
        }

        /**
         * Draw 5x5 QUBO Coupling Matrix on HTML5 Canvas
         */
        drawHeatmap(matrix) {
            const canvas = this.heatmapCanvas;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const w = canvas.width;
            const h = canvas.height;

            // Default 5x5 QUBO Coupling matrix (J_ij off-diagonal + h_i diagonal)
            const defaultMatrix = [
                [ 1.45, -0.62,  0.15, -0.05,  0.00],
                [-0.62,  2.10, -0.85,  0.20, -0.05],
                [ 0.15, -0.85,  1.80, -0.75,  0.10],
                [-0.05,  0.20, -0.75,  1.60, -0.55],
                [ 0.00, -0.05,  0.10, -0.55,  1.30]
            ];

            const mat = (matrix && Array.isArray(matrix) && matrix.length === 5) ? matrix : defaultMatrix;

            ctx.clearRect(0, 0, w, h);

            const n = 5;
            const pad = 24; // padding for labels
            const cellSize = (w - pad) / n;

            // Draw axis labels
            ctx.fillStyle = '#64748b';
            ctx.font = '10px Consolas';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            for (let i = 0; i < n; i++) {
                // Top column headers
                ctx.fillText(`Q${i}`, pad + i * cellSize + cellSize / 2, 12);
                // Left row headers
                ctx.fillText(`Q${i}`, 12, pad + i * cellSize + cellSize / 2);
            }

            // Draw cells
            for (let r = 0; r < n; r++) {
                for (let c = 0; c < n; c++) {
                    const val = mat[r][c];
                    const x = pad + c * cellSize;
                    const y = pad + r * cellSize;

                    // Color mapping: diagonal (positive queue bias) -> cyan/green;
                    // off-diagonal negative (synergy green wave) -> blue/purple
                    let fill;
                    if (r === c) {
                        // Diagonal h_i term
                        const alpha = Math.min(0.9, Math.max(0.2, val / 3.0));
                        fill = `rgba(0, 255, 136, ${alpha})`;
                    } else if (val < 0) {
                        // Synergistic coupling (green wave sync)
                        const alpha = Math.min(0.9, Math.max(0.15, Math.abs(val)));
                        fill = `rgba(0, 212, 255, ${alpha})`;
                    } else {
                        // Conflict / spillback penalty
                        const alpha = Math.min(0.9, Math.max(0.1, val));
                        fill = `rgba(239, 68, 68, ${alpha})`;
                    }

                    ctx.fillStyle = fill;
                    ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

                    // Cell value text
                    ctx.fillStyle = '#ffffff';
                    ctx.font = '8px Consolas';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(val.toFixed(2), x + cellSize / 2, y + cellSize / 2);
                }
            }
        }

        /**
         * Update quantum telemetry values
         * @param {Object} telemetry
         * @param {Array} intersections
         * @param {Object} stats
         */
        update(telemetry, intersections, stats) {
            const t = telemetry || {};

            // Update Metadata Items
            const updateText = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.textContent = val;
            };

            updateText('qaoaQubits', t.num_qubits || (intersections ? intersections.length : 5));
            updateText('qaoaGates', t.gates_count || 24);
            updateText('qaoaGamma', (t.gamma !== undefined) ? Number(t.gamma).toFixed(3) : '0.785');
            updateText('qaoaBeta', (t.beta !== undefined) ? Number(t.beta).toFixed(3) : '0.392');

            // Energy & Bitstring
            const bitstring = t.optimal_bitstring || '01010';
            updateText('qaoaBitstring', `|${bitstring}>`);
            updateText('qaoaEnergy', `E = ${(t.energy !== undefined ? Number(t.energy).toFixed(2) : '-18.42')}`);

            // Decode bitstring to intersection green wave assignments
            this.updateBitstringDecoder(bitstring, intersections);

            // Update Chart with probabilities
            if (t.probabilities && this.chart) {
                this.updateChartData(t.probabilities, bitstring);
            }

            // Update Heatmap if coupling matrix provided
            if (t.coupling_matrix) {
                this.drawHeatmap(t.coupling_matrix);
            }

            // Update Overall Stats
            if (stats) {
                updateText('statTotalVehicles', stats.total_vehicles || 0);
                updateText('statReduction', `${Math.round(stats.avg_wait_reduction || 45)}%`);
                updateText('statCO2', `${Math.round(stats.co2_saved_kg || 120)} kg`);
            }
        }

        /**
         * Decode bitstring (0 = NS Green, 1 = EW Green) per intersection
         */
        updateBitstringDecoder(bitstring, intersections) {
            const container = document.getElementById('bitstringDecoder');
            if (!container) return;

            let html = '';
            const bits = bitstring.split('');
            bits.forEach((bit, idx) => {
                const interName = (intersections && intersections[idx]) ? intersections[idx].name : `Node ${idx + 1}`;
                const phase = bit === '0' ? 'NS GREEN' : 'EW GREEN';
                const chipClass = bit === '0' ? 'ns' : 'ew';
                html += `<span class="decode-chip ${chipClass}">Q${idx} (${interName.substring(0, 10)}): ${phase}</span>`;
            });

            container.innerHTML = html;
        }

        updateChartData(probabilities, optimalBitstring) {
            if (!this.chart) return;

            let labels = [];
            let data = [];

            if (Array.isArray(probabilities)) {
                labels = probabilities.slice(0, 8).map(p => `|${p.state || p[0]}>`);
                data = probabilities.slice(0, 8).map(p => p.prob || p[1]);
            } else if (typeof probabilities === 'object') {
                const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]).slice(0, 8);
                labels = entries.map(e => `|${e[0]}>`);
                data = entries.map(e => e[1]);
            }

            if (labels.length > 0) {
                this.chart.data.labels = labels;
                this.chart.data.datasets[0].data = data;
                this.chart.data.datasets[0].backgroundColor = labels.map(l =>
                    l === `|${optimalBitstring}>` ? 'rgba(0, 255, 136, 0.85)' : 'rgba(0, 212, 255, 0.45)'
                );
                this.chart.update('none');
            }
        }
    }

    // Export module
    window.QuantumPanel = QuantumPanel;
})();
