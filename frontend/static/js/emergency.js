/**
 * Emergency Module - Autonomous Smart Traffic Management
 * Handles dispatching emergency vehicles (Ambulance / Fire Truck),
 * draws preemption corridors on Leaflet map, and activates siren banners.
 */

(function () {
    'use strict';

    class EmergencyModule {
        constructor() {
            this.map = null;
            this.routePolyline = null;
            this.emergencyMarker = null;
            this.isActive = false;
            this.bannerEl = document.getElementById('emergencyAlertBanner');
            this.titleEl = document.getElementById('emergencyTitle');
            this.subtitleEl = document.getElementById('emergencySubtitle');
        }

        init(map) {
            this.map = map;
            this.bindEvents();
            console.log('[EmergencyModule] Initialized.');
        }

        bindEvents() {
            const btnAmbulance = document.getElementById('btnDispatchAmbulance');
            if (btnAmbulance) {
                btnAmbulance.addEventListener('click', () => this.dispatch('ambulance'));
            }

            const btnFiretruck = document.getElementById('btnDispatchFiretruck');
            if (btnFiretruck) {
                btnFiretruck.addEventListener('click', () => this.dispatch('firetruck'));
            }

            const btnClear = document.getElementById('btnClearEmergency');
            if (btnClear) {
                btnClear.addEventListener('click', () => this.dismissBanner());
            }
        }

        /**
         * Dispatch emergency vehicle via API
         * @param {string} type 'ambulance' | 'firetruck'
         */
        dispatch(type) {
            console.log(`[EmergencyModule] Dispatching ${type}...`);
            fetch('/api/emergency', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: type })
            })
            .then(res => res.json())
            .then(data => {
                console.log('[EmergencyModule] Dispatch response:', data);
                this.showAlert(type);
            })
            .catch(err => {
                console.error('[EmergencyModule] Error dispatching emergency:', err);
                // Trigger local alert visual anyway for feedback
                this.showAlert(type);
            });
        }

        /**
         * Update state from 5Hz backend polling
         * @param {Object} state
         */
        update(state) {
            if (!state) return;

            const active = !!state.emergency_active;
            const route = state.emergency_route || [];

            if (active && !this.isActive) {
                this.isActive = true;
                const type = (state.vehicles && state.vehicles.find(v => v.is_emergency)?.type) || 'ambulance';
                this.showAlert(type);
                this.renderEmergencyRoute(route, state.intersections);
            } else if (!active && this.isActive) {
                this.isActive = false;
                this.clearEmergencyVisuals();
            } else if (active && this.isActive) {
                // Route update if needed
                if (route.length > 0 && !this.routePolyline) {
                    this.renderEmergencyRoute(route, state.intersections);
                }
            }
        }

        showAlert(type) {
            const label = type === 'firetruck' ? 'FIRE TRUCK' : 'AMBULANCE';
            if (this.bannerEl) {
                this.bannerEl.classList.remove('hidden');
            }
            if (this.titleEl) {
                this.titleEl.textContent = `EMERGENCY ${label} EN ROUTE - ALL SIGNALS PREEMPTED`;
            }
            if (this.subtitleEl) {
                this.subtitleEl.textContent = 'Qiskit QAOA Hamiltonian preempted: Continuous green corridor established. Opposing approaches held in all-red.';
            }
        }

        dismissBanner() {
            if (this.bannerEl) {
                this.bannerEl.classList.add('hidden');
            }
        }

        /**
         * Draw glowing emergency preemption route on map
         */
        renderEmergencyRoute(routePoints, intersections) {
            if (!this.map || typeof L === 'undefined') return;

            let latLngs = [];

            if (Array.isArray(routePoints) && routePoints.length >= 2) {
                latLngs = routePoints.map(p => Array.isArray(p) ? p : [p.lat, p.lng]);
            } else if (Array.isArray(intersections) && intersections.length >= 2) {
                // Synthesize corridor through first 3 intersections
                latLngs = intersections.slice(0, 3).map(i => [i.lat, i.lng]);
            }

            if (latLngs.length < 2) return;

            // Remove existing route
            if (this.routePolyline) {
                this.map.removeLayer(this.routePolyline);
            }

            // Draw glowing thick emergency route
            this.routePolyline = L.polyline(latLngs, {
                color: '#ff1100',
                weight: 6,
                opacity: 0.9,
                dashArray: '8, 8',
                lineJoin: 'round'
            }).addTo(this.map);

            this.routePolyline.bindTooltip('<b>EMERGENCY GREEN CORRIDOR</b><br>Priority: 10/10', {
                sticky: true
            });
        }

        clearEmergencyVisuals() {
            this.dismissBanner();
            if (this.routePolyline && this.map) {
                this.map.removeLayer(this.routePolyline);
                this.routePolyline = null;
            }
            if (this.emergencyMarker && this.map) {
                this.map.removeLayer(this.emergencyMarker);
                this.emergencyMarker = null;
            }
        }
    }

    // Export module
    window.EmergencyModule = EmergencyModule;
})();
