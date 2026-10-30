/**
 * Main Application Orchestrator - Autonomous Smart Traffic Management
 * Coordinates state polling at ~5Hz (200ms), dispatches updates to
 * MapRenderer, TrafficLights, VehicleLayer, EmergencyModule, QuantumPanel, DLPanel.
 * Manages user interactions: city switching, time slots, mode toggle, playback, and surges.
 */

(function () {
    'use strict';

    class App {
        constructor() {
            this.mapRenderer = null;
            this.trafficLights = null;
            this.vehicleLayer = null;
            this.emergencyModule = null;
            this.quantumPanel = null;
            this.dlPanel = null;

            this.pollIntervalMs = 200; // ~5Hz
            this.isPolling = false;
            this.pollTimer = null;
            this.currentCity = 'bhopal';
            this.currentMode = 'quantum';
            this.isPlaying = true;
            this.playbackSpeed = 1.0;
            this.consecutiveErrors = 0;
        }

        init() {
            console.log('[App] Initializing Command Center Frontend...');

            // 1. Initialize Sub-modules
            this.mapRenderer = new window.MapRenderer();
            this.mapRenderer.init('map');

            const leafletMap = this.mapRenderer.getMap();

            this.trafficLights = new window.TrafficLights();
            this.trafficLights.init(leafletMap);

            this.vehicleLayer = new window.VehicleLayer();
            this.vehicleLayer.init(leafletMap);

            this.emergencyModule = new window.EmergencyModule();
            this.emergencyModule.init(leafletMap);

            this.quantumPanel = new window.QuantumPanel();
            this.quantumPanel.init();

            this.dlPanel = new window.DLPanel();
            this.dlPanel.init();

            // Export instances for debugging
            window.mapRenderer = this.mapRenderer;
            window.trafficLights = this.trafficLights;
            window.vehicleLayer = this.vehicleLayer;
            window.emergencyModule = this.emergencyModule;
            window.quantumPanel = this.quantumPanel;
            window.dlPanel = this.dlPanel;

            // 2. Bind DOM Event Listeners
            this.bindEvents();

            // 3. Start 5Hz Telemetry Polling Loop
            this.startPolling();

            console.log('[App] System ready and polling.');
        }

        bindEvents() {
            // City Selector Buttons
            const cityButtons = document.querySelectorAll('.city-btn');
            cityButtons.forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const cityId = e.currentTarget.getAttribute('data-city');
                    this.switchCity(cityId);
                });
            });

            // Time Slot Selector
            const timeSlotSelect = document.getElementById('timeSlotSelect');
            if (timeSlotSelect) {
                timeSlotSelect.addEventListener('change', (e) => {
                    const slotId = e.target.value;
                    this.switchTimeSlot(slotId);
                });
            }

            // Mode Toggle (Quantum QAOA vs Classical Fixed)
            const btnMode = document.getElementById('btnModeToggle');
            if (btnMode) {
                btnMode.addEventListener('click', () => this.toggleMode());
            }

            // Playback Controls (Play / Pause & Speed)
            const btnPlayPause = document.getElementById('btnPlayPause');
            if (btnPlayPause) {
                btnPlayPause.addEventListener('click', () => this.togglePlayback());
            }

            const speedSelect = document.getElementById('speedSelect');
            if (speedSelect) {
                speedSelect.addEventListener('change', (e) => {
                    this.playbackSpeed = parseFloat(e.target.value) || 1.0;
                    this.sendPlaybackState();
                });
            }

            // Traffic Surge Button (+16 Vehicles)
            const btnSurge = document.getElementById('btnTrafficSurge');
            if (btnSurge) {
                btnSurge.addEventListener('click', () => this.triggerTrafficSurge());
            }

            // Google Maps Key Modal
            this.bindGoogleKeyModal();
        }

        bindGoogleKeyModal() {
            const btnOpen = document.getElementById('btnGoogleKeyModal');
            const btnClose = document.getElementById('btnCloseKeyModal');
            const modal = document.getElementById('googleKeyModal');
            const btnSave = document.getElementById('btnSaveGoogleKey');
            const inputKey = document.getElementById('inputGoogleKey');

            if (btnOpen && modal) {
                btnOpen.addEventListener('click', () => modal.classList.remove('hidden'));
            }
            if (btnClose && modal) {
                btnClose.addEventListener('click', () => modal.classList.add('hidden'));
            }
            if (btnSave && inputKey) {
                btnSave.addEventListener('click', () => {
                    const key = inputKey.value.trim();
                    if (!key) return;
                    fetch('/api/google_key', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ api_key: key })
                    })
                    .then(res => res.json())
                    .then(() => {
                        alert('Google Maps API Key saved successfully.');
                        if (modal) modal.classList.add('hidden');
                    })
                    .catch(err => console.error('[App] Error saving Google Key:', err));
                });
            }
        }

        /**
         * 5Hz Polling loop to fetch /api/state
         */
        startPolling() {
            this.isPolling = true;

            const poll = () => {
                if (!this.isPolling) return;

                fetch('/api/state')
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP ${response.status}`);
                        return response.json();
                    })
                    .then(state => {
                        this.consecutiveErrors = 0;
                        this.updateConnectionStatus(true);
                        this.distributeState(state);
                        this.pollTimer = setTimeout(poll, this.pollIntervalMs);
                    })
                    .catch(err => {
                        this.consecutiveErrors++;
                        this.updateConnectionStatus(false);
                        // Backoff slightly on errors
                        const backoff = Math.min(2000, 500 + this.consecutiveErrors * 300);
                        this.pollTimer = setTimeout(poll, backoff);
                    });
            };

            poll();
        }

        /**
         * Distribute backend telemetry to all modules
         * @param {Object} state
         */
        distributeState(state) {
            if (!state) return;

            // Update top bar city and time slot if changed remotely
            this.syncRemoteStateControls(state);

            // 1. Update Leaflet Map
            if (this.mapRenderer) {
                this.mapRenderer.update(state);
            }

            // 2. Update Traffic Lights (Markers + Sidebar)
            if (this.trafficLights && state.intersections) {
                this.trafficLights.update(state.intersections);
            }

            // 3. Update Vehicle Markers and Congestion Breakdown
            if (this.vehicleLayer) {
                this.vehicleLayer.update(state.vehicles || [], state.intersections || []);
            }

            // 4. Update Emergency Preemption Corridors and Sirens
            if (this.emergencyModule) {
                this.emergencyModule.update(state);
            }

            // 5. Update Quantum QAOA Telemetry and QUBO Heatmap
            if (this.quantumPanel) {
                this.quantumPanel.update(state.quantum_telemetry, state.intersections, state.stats);
            }

            // 6. Update Deep Learning Forecasts and YOLO Mix
            if (this.dlPanel) {
                this.dlPanel.update(state.dl_predictions, state.intersections);
            }
        }

        /**
         * Synchronize top bar active buttons with backend state
         */
        syncRemoteStateControls(state) {
            if (state.city && state.city !== this.currentCity) {
                this.currentCity = state.city;
                const cityBtns = document.querySelectorAll('.city-btn');
                cityBtns.forEach(b => {
                    if (b.getAttribute('data-city') === state.city) {
                        b.classList.add('active');
                    } else {
                        b.classList.remove('active');
                    }
                });
            }

            if (state.time_slot && state.time_slot.id) {
                const select = document.getElementById('timeSlotSelect');
                if (select && select.value !== state.time_slot.id) {
                    select.value = state.time_slot.id;
                }
            }

            if (state.stats && state.stats.mode && state.stats.mode !== this.currentMode) {
                this.currentMode = state.stats.mode;
                this.updateModeUI(this.currentMode);
            }
        }

        /**
         * Switch active city
         * @param {string} cityId
         */
        switchCity(cityId) {
            if (!cityId || cityId === this.currentCity) return;
            console.log(`[App] Switching city to: ${cityId}`);

            this.currentCity = cityId;

            // Update UI buttons immediately
            const cityBtns = document.querySelectorAll('.city-btn');
            cityBtns.forEach(b => {
                b.classList.toggle('active', b.getAttribute('data-city') === cityId);
            });

            // Clear old markers before flyTo
            if (this.trafficLights) this.trafficLights.clearMarkers();
            if (this.vehicleLayer) this.vehicleLayer.clearMarkers();

            fetch('/api/city', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ city_id: cityId })
            })
            .then(res => res.json())
            .then(data => {
                console.log('[App] City switched response:', data);
                if (this.mapRenderer) {
                    this.mapRenderer.setCity(cityId, data.intersections || []);
                }
            })
            .catch(err => console.error('[App] Error switching city:', err));
        }

        /**
         * Switch 3-hour time slot
         * @param {string} slotId
         */
        switchTimeSlot(slotId) {
            console.log(`[App] Switching time slot to: ${slotId}`);
            fetch('/api/time_slot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ slot_id: slotId })
            })
            .then(res => res.json())
            .then(data => console.log('[App] Time slot switched response:', data))
            .catch(err => console.error('[App] Error switching time slot:', err));
        }

        /**
         * Toggle between Quantum QAOA and Classical Fixed timing
         */
        toggleMode() {
            const nextMode = (this.currentMode === 'quantum') ? 'classical' : 'quantum';
            this.currentMode = nextMode;
            this.updateModeUI(nextMode);

            fetch('/api/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode: nextMode })
            })
            .then(res => res.json())
            .then(data => console.log('[App] Mode updated:', data))
            .catch(err => console.error('[App] Error updating mode:', err));
        }

        updateModeUI(mode) {
            const badge = document.getElementById('systemModeBadge');
            const toggleLabel = document.getElementById('modeToggleLabel');

            if (mode === 'quantum') {
                if (badge) {
                    badge.innerHTML = '<i class="fa-solid fa-atom"></i> Mode: Quantum QAOA';
                    badge.className = 'badge badge-quantum';
                }
                if (toggleLabel) toggleLabel.textContent = 'Classical';
            } else {
                if (badge) {
                    badge.innerHTML = '<i class="fa-solid fa-stopwatch"></i> Mode: Classical Fixed';
                    badge.className = 'badge badge-status';
                }
                if (toggleLabel) toggleLabel.textContent = 'Quantum';
            }
        }

        /**
         * Playback play/pause toggle
         */
        togglePlayback() {
            this.isPlaying = !this.isPlaying;
            const btn = document.getElementById('btnPlayPause');
            const label = document.getElementById('playbackStatusLabel');

            if (btn && label) {
                if (this.isPlaying) {
                    btn.innerHTML = '<i class="fa-solid fa-pause"></i> <span id="playbackStatusLabel">Pause</span>';
                    btn.classList.add('primary');
                } else {
                    btn.innerHTML = '<i class="fa-solid fa-play"></i> <span id="playbackStatusLabel">Resume</span>';
                    btn.classList.remove('primary');
                }
            }

            this.sendPlaybackState();
        }

        sendPlaybackState() {
            fetch('/api/playback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    playing: this.isPlaying,
                    speed: this.playbackSpeed
                })
            })
            .then(res => res.json())
            .catch(err => console.error('[App] Error updating playback:', err));
        }

        /**
         * Trigger traffic burst surge (+16 vehicles)
         */
        triggerTrafficSurge() {
            const btn = document.getElementById('btnTrafficSurge');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Injecting Burst...';
            }

            fetch('/api/burst', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ count: 16 })
            })
            .then(res => res.json())
            .then(data => {
                console.log('[App] Surge injected:', data);
                setTimeout(() => {
                    if (btn) {
                        btn.disabled = false;
                        btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Inject Traffic Surge (+16 Vehicles)';
                    }
                }, 600);
            })
            .catch(err => {
                console.error('[App] Error injecting surge:', err);
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Inject Traffic Surge (+16 Vehicles)';
                }
            });
        }

        /**
         * Connection status badge update
         */
        updateConnectionStatus(isOnline) {
            const statusEl = document.getElementById('connectionStatus');
            if (!statusEl) return;

            if (isOnline) {
                statusEl.className = 'status-indicator online';
                statusEl.innerHTML = '<span class="status-dot"></span> <span class="status-text">5Hz LIVE</span>';
            } else {
                statusEl.className = 'status-indicator offline';
                statusEl.innerHTML = '<span class="status-dot" style="background:#ff4444;box-shadow:none;"></span> <span class="status-text" style="color:#ff8888;">RETRYING</span>';
            }
        }
    }

    // Initialize application when DOM is ready
    window.App = App;
    document.addEventListener('DOMContentLoaded', () => {
        window.appInstance = new App();
        window.appInstance.init();
    });
})();
