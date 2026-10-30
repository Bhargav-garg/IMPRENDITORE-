/**
 * Map Renderer Module - Autonomous Smart Traffic Management
 * Initializes Leaflet map, renders CartoDB Dark & OSM tile layers,
 * renders multi-lane road polylines connecting intersections,
 * handles city flyTo animations and intersection focus pills.
 */

(function () {
    'use strict';

    const CITY_COORDINATES = {
        bhopal: [23.2599, 77.4126],
        gwalior: [26.2183, 78.1828],
        newdelhi: [28.6139, 77.2090],
        bangalore: [12.9716, 77.5946],
        chandigarh: [30.7333, 76.7794]
    };

    class MapRenderer {
        constructor() {
            this.map = null;
            this.currentCity = 'bhopal';
            this.intersections = [];
            this.roadLayers = [];
            this.darkLayer = null;
            this.osmLayer = null;
            this.activeTile = 'dark';
        }

        /**
         * Initialize Leaflet map in target DOM container
         * @param {string} containerId default 'map'
         */
        init(containerId = 'map') {
            if (typeof L === 'undefined') {
                console.error('[MapRenderer] Leaflet library not loaded.');
                return;
            }

            const initialCenter = CITY_COORDINATES.bhopal;
            this.map = L.map(containerId, {
                center: initialCenter,
                zoom: 14,
                zoomControl: false,
                attributionControl: false
            });

            // Position zoom control at bottom-right
            L.control.zoom({ position: 'bottomright' }).addTo(this.map);

            // 1. CartoDB Dark All Basemap (Free, high-contrast command center look)
            this.darkLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                maxZoom: 19,
                subdomains: 'abcd'
            });

            // 2. OpenStreetMap Standard Basemap (Free alternative)
            this.osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19
            });

            // Add default dark layer
            this.darkLayer.addTo(this.map);

            this.bindLayerControls();
            console.log('[MapRenderer] Leaflet map initialized successfully.');
        }

        bindLayerControls() {
            const btnDark = document.getElementById('btnLayerDark');
            const btnOsm = document.getElementById('btnLayerOsm');
            const btnFit = document.getElementById('btnFitBounds');

            if (btnDark) {
                btnDark.addEventListener('click', () => {
                    if (this.activeTile !== 'dark') {
                        this.map.removeLayer(this.osmLayer);
                        this.darkLayer.addTo(this.map);
                        this.activeTile = 'dark';
                        btnDark.classList.add('active');
                        if (btnOsm) btnOsm.classList.remove('active');
                    }
                });
            }

            if (btnOsm) {
                btnOsm.addEventListener('click', () => {
                    if (this.activeTile !== 'osm') {
                        this.map.removeLayer(this.darkLayer);
                        this.osmLayer.addTo(this.map);
                        this.activeTile = 'osm';
                        btnOsm.classList.add('active');
                        if (btnDark) btnDark.classList.remove('active');
                    }
                });
            }

            if (btnFit) {
                btnFit.addEventListener('click', () => this.fitAllIntersections());
            }
        }

        /**
         * Change city and reposition map
         * @param {string} cityId
         * @param {Array} intersections
         */
        setCity(cityId, intersections = []) {
            this.currentCity = cityId.toLowerCase();
            this.intersections = intersections;

            const targetCenter = CITY_COORDINATES[this.currentCity] || (
                intersections.length > 0 ? [intersections[0].lat, intersections[0].lng] : [23.2599, 77.4126]
            );

            if (this.map) {
                this.map.flyTo(targetCenter, 14, {
                    animate: true,
                    duration: 1.2
                });
            }

            this.rebuildRoadNetwork(intersections);
            this.updateIntersectionPills(intersections);

            // Auto-fit bounds if intersections exist
            if (intersections.length > 0) {
                setTimeout(() => this.fitAllIntersections(), 1300);
            }
        }

        /**
         * Update map with current state payload
         * @param {Object} state
         */
        update(state) {
            if (!state) return;

            // Check if city changed
            if (state.city && state.city.toLowerCase() !== this.currentCity) {
                this.setCity(state.city, state.intersections || []);
            } else if (state.intersections && this.intersections.length === 0) {
                this.intersections = state.intersections;
                this.rebuildRoadNetwork(state.intersections);
                this.updateIntersectionPills(state.intersections);
                this.fitAllIntersections();
            }
        }

        /**
         * Draw asphalt roads and dashed lane markings between adjacent intersections
         */
        rebuildRoadNetwork(intersections) {
            if (!this.map || typeof L === 'undefined') return;

            // Clear old road polylines
            this.roadLayers.forEach(layer => this.map.removeLayer(layer));
            this.roadLayers = [];

            if (!Array.isArray(intersections) || intersections.length < 2) return;

            const points = intersections.map(i => [i.lat, i.lng]);

            // Draw base road surface (dark asphalt)
            const roadBase = L.polyline(points, {
                color: '#1e293b',
                weight: 10,
                opacity: 0.85,
                lineCap: 'round',
                lineJoin: 'round'
            }).addTo(this.map);
            this.roadLayers.push(roadBase);

            // Draw inner lane divider (dashed road marking)
            const laneDivider = L.polyline(points, {
                color: '#64748b',
                weight: 1.5,
                dashArray: '8, 10',
                opacity: 0.75,
                lineCap: 'round',
                lineJoin: 'round'
            }).addTo(this.map);
            this.roadLayers.push(laneDivider);
        }

        /**
         * Auto-fit bounds to enclose all intersections with comfortable padding
         */
        fitAllIntersections() {
            if (!this.map || !this.intersections || this.intersections.length === 0) return;

            const bounds = L.latLngBounds(this.intersections.map(i => [i.lat, i.lng]));
            this.map.fitBounds(bounds, {
                padding: [50, 50],
                maxZoom: 16
            });
        }

        /**
         * Focus smoothly on a specific intersection
         */
        zoomTo(intersection) {
            if (!this.map || !intersection) return;
            this.map.flyTo([intersection.lat, intersection.lng], 16, {
                animate: true,
                duration: 0.8
            });
        }

        /**
         * Render quick-jump pills on map floating toolbar
         */
        updateIntersectionPills(intersections) {
            const container = document.getElementById('intersectionPills');
            if (!container) return;

            let html = '';
            intersections.forEach((inter, idx) => {
                html += `<button class="intersection-pill" data-idx="${idx}">${inter.name}</button>`;
            });

            container.innerHTML = html;

            // Attach click listener
            const pills = container.querySelectorAll('.intersection-pill');
            pills.forEach(pill => {
                pill.addEventListener('click', (e) => {
                    const idx = parseInt(e.currentTarget.getAttribute('data-idx'), 10);
                    if (this.intersections[idx]) {
                        this.zoomTo(this.intersections[idx]);
                    }
                });
            });
        }

        getMap() {
            return this.map;
        }
    }

    // Export module
    window.MapRenderer = MapRenderer;
})();
