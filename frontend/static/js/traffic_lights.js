/**
 * Traffic Lights Module - Autonomous Smart Traffic Management
 * Visualizes dual-phase traffic light states (NS / EW) with glow effects,
 * countdown timers, and directional queue counters on the map and in the sidebar.
 */

(function () {
    'use strict';

    class TrafficLights {
        constructor() {
            this.map = null;
            this.markers = new Map(); // id -> L.marker
            this.intersections = [];
            this.sidebarContainer = document.getElementById('trafficLightsList');
            this.waitTimeContainer = document.getElementById('waitTimeCards');
        }

        init(map) {
            this.map = map;
            console.log('[TrafficLights] Initialized with map reference.');
        }

        /**
         * Update traffic lights from backend intersections state array
         * @param {Array} intersections
         */
        update(intersections) {
            if (!intersections || !Array.isArray(intersections)) return;
            this.intersections = intersections;

            this.updateMapMarkers(intersections);
            this.updateSidebarList(intersections);
            this.updateWaitTimeCards(intersections);
        }

        /**
         * Update or create Leaflet DivIcon markers on the map
         */
        updateMapMarkers(intersections) {
            if (!this.map || typeof L === 'undefined') return;

            const currentIds = new Set();

            intersections.forEach((inter) => {
                const id = inter.id !== undefined ? inter.id : inter.name;
                currentIds.add(id);

                const iconHtml = this.generateMapMarkerHtml(inter);
                const customIcon = L.divIcon({
                    html: iconHtml,
                    className: 'custom-tl-leaflet-icon',
                    iconSize: [82, 70],
                    iconAnchor: [41, 35]
                });

                if (this.markers.has(id)) {
                    const marker = this.markers.get(id);
                    marker.setLatLng([inter.lat, inter.lng]);
                    marker.setIcon(customIcon);
                } else {
                    const marker = L.marker([inter.lat, inter.lng], {
                        icon: customIcon,
                        zIndexOffset: 1000
                    }).addTo(this.map);

                    marker.bindTooltip(`<b>${inter.name}</b><br>State: ${inter.light_state}<br>Wait: ${inter.quantum_wait_sec || 20}s`, {
                        direction: 'top',
                        offset: [0, -25]
                    });

                    this.markers.set(id, marker);
                }
            });

            // Remove stale markers for previous city
            for (const [id, marker] of this.markers.entries()) {
                if (!currentIds.has(id)) {
                    this.map.removeLayer(marker);
                    this.markers.delete(id);
                }
            }
        }

        /**
         * Generate HTML string for map DivIcon
         */
        generateMapMarkerHtml(inter) {
            const state = inter.light_state || 'ALL_RED';
            const timer = Math.max(0, Math.round(inter.timer_sec || 0));

            let nsLight = 'red';
            let ewLight = 'red';

            if (state === 'NS_GREEN') {
                nsLight = 'green';
                ewLight = 'red';
            } else if (state === 'EW_GREEN') {
                nsLight = 'red';
                ewLight = 'green';
            } else if (state === 'NS_YELLOW') {
                nsLight = 'yellow';
                ewLight = 'red';
            } else if (state === 'EW_YELLOW') {
                nsLight = 'red';
                ewLight = 'yellow';
            } else if (state === 'ALL_RED') {
                nsLight = 'red';
                ewLight = 'red';
            }

            return `
                <div class="map-tl-marker">
                    <div class="map-tl-title" title="${inter.name}">${inter.name}</div>
                    <div class="map-tl-lights">
                        <div class="map-tl-light-col">
                            <span class="map-tl-dir">NS</span>
                            <div class="map-light-circle ${nsLight}"></div>
                        </div>
                        <div class="map-tl-light-col">
                            <span class="map-tl-dir">EW</span>
                            <div class="map-light-circle ${ewLight}"></div>
                        </div>
                    </div>
                    <div class="map-tl-countdown">${timer}s</div>
                </div>
            `;
        }

        /**
         * Update sidebar list with detailed cards
         */
        updateSidebarList(intersections) {
            if (!this.sidebarContainer) return;

            let html = '';
            intersections.forEach((inter) => {
                const state = inter.light_state || 'ALL_RED';
                const timer = Math.max(0, Math.round(inter.timer_sec || 0));
                const queues = inter.queues || { N: 0, S: 0, E: 0, W: 0 };

                let nsRedLit = false, nsYellowLit = false, nsGreenLit = false;
                let ewRedLit = false, ewYellowLit = false, ewGreenLit = false;
                let badgeClass = 'phase-all-red';
                let badgeText = 'ALL RED';

                if (state === 'NS_GREEN') {
                    nsGreenLit = true;
                    ewRedLit = true;
                    badgeClass = 'phase-ns-green';
                    badgeText = 'NS GREEN';
                } else if (state === 'EW_GREEN') {
                    nsRedLit = true;
                    ewGreenLit = true;
                    badgeClass = 'phase-ew-green';
                    badgeText = 'EW GREEN';
                } else if (state === 'NS_YELLOW') {
                    nsYellowLit = true;
                    ewRedLit = true;
                    badgeClass = 'phase-yellow';
                    badgeText = 'NS YELLOW';
                } else if (state === 'EW_YELLOW') {
                    nsRedLit = true;
                    ewYellowLit = true;
                    badgeClass = 'phase-yellow';
                    badgeText = 'EW YELLOW';
                } else if (state === 'ALL_RED') {
                    nsRedLit = true;
                    ewRedLit = true;
                    badgeClass = 'phase-all-red';
                    badgeText = 'PREEMPTION RED';
                }

                html += `
                    <div class="traffic-light-card" data-id="${inter.id}">
                        <div class="tl-card-header">
                            <span class="tl-name">${inter.name}</span>
                            <span class="tl-phase-badge ${badgeClass}">${badgeText}</span>
                        </div>
                        <div class="tl-graphic-row">
                            <!-- North-South Direction Unit -->
                            <div class="tl-direction-unit">
                                <span class="tl-dir-label">N - S</span>
                                <div class="tl-housing">
                                    <div class="tl-circle red ${nsRedLit ? 'lit' : ''}"></div>
                                    <div class="tl-circle yellow ${nsYellowLit ? 'lit' : ''}"></div>
                                    <div class="tl-circle green ${nsGreenLit ? 'lit' : ''}"></div>
                                </div>
                            </div>

                            <!-- Countdown Timer Display -->
                            <div class="tl-timer-display">
                                <span class="tl-countdown">${timer}s</span>
                                <span class="tl-countdown-sub">REMAINING</span>
                            </div>

                            <!-- East-West Direction Unit -->
                            <div class="tl-direction-unit">
                                <span class="tl-dir-label">E - W</span>
                                <div class="tl-housing">
                                    <div class="tl-circle red ${ewRedLit ? 'lit' : ''}"></div>
                                    <div class="tl-circle yellow ${ewYellowLit ? 'lit' : ''}"></div>
                                    <div class="tl-circle green ${ewGreenLit ? 'lit' : ''}"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Queue Lengths -->
                        <div class="tl-queues-row">
                            <div class="queue-pill"><span class="q-dir">N:</span><span class="q-val">${queues.N || 0}</span></div>
                            <div class="queue-pill"><span class="q-dir">S:</span><span class="q-val">${queues.S || 0}</span></div>
                            <div class="queue-pill"><span class="q-dir">E:</span><span class="q-val">${queues.E || 0}</span></div>
                            <div class="queue-pill"><span class="q-dir">W:</span><span class="q-val">${queues.W || 0}</span></div>
                        </div>
                    </div>
                `;
            });

            this.sidebarContainer.innerHTML = html;
        }

        /**
         * Update Wait Time Comparison Cards (Fixed vs Quantum)
         */
        updateWaitTimeCards(intersections) {
            if (!this.waitTimeContainer) return;

            let html = '';
            intersections.forEach((inter) => {
                const fixedWait = inter.earlier_wait_sec || 65;
                const quantumWait = inter.quantum_wait_sec || 20;
                const savedSec = Math.max(0, fixedWait - quantumWait);
                const reductionPct = fixedWait > 0 ? Math.round((savedSec / fixedWait) * 100) : 0;

                html += `
                    <div class="wait-card">
                        <div class="wait-card-header">
                            <span class="wait-card-name">${inter.name}</span>
                            <span class="wait-savings-tag">-${reductionPct}% Wait</span>
                        </div>
                        <div class="wait-comparison-row">
                            <div class="wait-col fixed">
                                <span class="w-label">Earlier (Fixed)</span>
                                <span class="w-val">${fixedWait}s</span>
                            </div>
                            <div class="wait-col quantum">
                                <span class="w-label">Current (QAOA)</span>
                                <span class="w-val">${quantumWait}s</span>
                            </div>
                            <div class="wait-col diff">
                                <span class="w-label">Saved</span>
                                <span class="w-val">-${savedSec}s</span>
                            </div>
                        </div>
                    </div>
                `;
            });

            this.waitTimeContainer.innerHTML = html;
        }

        getMarkers() {
            return this.markers;
        }

        clearMarkers() {
            if (this.map) {
                for (const marker of this.markers.values()) {
                    this.map.removeLayer(marker);
                }
            }
            this.markers.clear();
        }
    }

    // Export module
    window.TrafficLights = TrafficLights;
})();
