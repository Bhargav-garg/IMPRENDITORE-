/**
 * Vehicle Layer Module - Autonomous Smart Traffic Management
 * Handles real-time vehicle rendering using Leaflet CircleMarkers,
 * tracks vehicle counts by classification type, and calculates weighted congestion.
 */

(function () {
    'use strict';

    const VEHICLE_STYLES = {
        car: { color: '#3498db', fillColor: '#3498db', radius: 5.5, weight: 1.0 },
        bus: { color: '#e67e22', fillColor: '#e67e22', radius: 8.0, weight: 3.0 },
        truck: { color: '#e74c3c', fillColor: '#e74c3c', radius: 9.0, weight: 3.5 },
        bike: { color: '#2ecc71', fillColor: '#2ecc71', radius: 4.0, weight: 0.5 },
        auto: { color: '#f1c40f', fillColor: '#f1c40f', radius: 5.0, weight: 0.8 },
        ambulance: { color: '#ff0000', fillColor: '#ffffff', radius: 9.5, weight: 10.0 },
        firetruck: { color: '#ff4500', fillColor: '#ffff00', radius: 10.0, weight: 9.0 }
    };

    class VehicleLayer {
        constructor() {
            this.map = null;
            this.markers = new Map(); // id -> L.circleMarker
            this.counts = { car: 0, bus: 0, truck: 0, bike: 0, auto: 0, ambulance: 0, firetruck: 0 };
        }

        init(map) {
            this.map = map;
            console.log('[VehicleLayer] Initialized on map.');
        }

        /**
         * Update vehicles on map and update sidebar detection panel
         * @param {Array} vehicles
         * @param {Array} intersections
         */
        update(vehicles, intersections) {
            if (!this.map || typeof L === 'undefined') return;
            const vehicleList = Array.isArray(vehicles) ? vehicles : [];

            this.updateMapMarkers(vehicleList);
            this.updateCountsAndCongestion(vehicleList, intersections);
        }

        /**
         * Render or reposition Leaflet CircleMarkers
         */
        updateMapMarkers(vehicles) {
            const currentIds = new Set();

            vehicles.forEach((v) => {
                const id = v.id || `v_${Math.random()}`;
                currentIds.add(id);

                const type = (v.type || 'car').toLowerCase();
                const style = VEHICLE_STYLES[type] || VEHICLE_STYLES.car;
                const isEmergency = v.is_emergency || type === 'ambulance' || type === 'firetruck';

                const latLng = [v.lat, v.lng];

                if (this.markers.has(id)) {
                    const marker = this.markers.get(id);
                    marker.setLatLng(latLng);
                } else {
                    const marker = L.circleMarker(latLng, {
                        radius: style.radius,
                        color: style.color,
                        weight: isEmergency ? 3 : 1.5,
                        opacity: 0.95,
                        fillColor: style.fillColor,
                        fillOpacity: 0.85,
                        zIndexOffset: isEmergency ? 2000 : 500
                    }).addTo(this.map);

                    const emergencyTag = isEmergency ? '<br><span style="color:#ff4444;font-weight:bold;">[EMERGENCY VEHICLE]</span>' : '';
                    marker.bindTooltip(`
                        <b>Vehicle:</b> ${id}<br>
                        <b>Type:</b> ${type.toUpperCase()}<br>
                        <b>Speed:</b> ${Math.round(v.speed_kmh || 0)} km/h
                        ${emergencyTag}
                    `, { direction: 'top', offset: [0, -5] });

                    this.markers.set(id, marker);
                }
            });

            // Remove markers of departed vehicles
            for (const [id, marker] of this.markers.entries()) {
                if (!currentIds.has(id)) {
                    this.map.removeLayer(marker);
                    this.markers.delete(id);
                }
            }
        }

        /**
         * Compute vehicle counts, weighted congestion index, and update DOM elements
         */
        updateCountsAndCongestion(vehicles, intersections) {
            const counts = { car: 0, bus: 0, truck: 0, bike: 0, auto: 0, ambulance: 0, firetruck: 0 };

            // Aggregate counts from vehicle objects or fallback to intersections vehicle_detected
            if (vehicles.length > 0) {
                vehicles.forEach((v) => {
                    const t = (v.type || 'car').toLowerCase();
                    if (counts[t] !== undefined) {
                        counts[t]++;
                    } else {
                        counts.car++;
                    }
                });
            } else if (Array.isArray(intersections)) {
                intersections.forEach((inter) => {
                    const vd = inter.vehicles_detected || {};
                    counts.car += vd.car || 0;
                    counts.bus += vd.bus || 0;
                    counts.truck += vd.truck || 0;
                    counts.bike += vd.bike || 0;
                    counts.auto += vd.auto || 0;
                });
            }

            this.counts = counts;
            const total = counts.car + counts.bus + counts.truck + counts.bike + counts.auto;

            // Update Count Labels
            const elTotal = document.getElementById('totalVehiclesBadge');
            if (elTotal) elTotal.textContent = `${total} Vehicles`;

            const updateEl = (id, val) => {
                const el = document.getElementById(id);
                if (el) el.textContent = val;
            };

            updateEl('countCar', counts.car);
            updateEl('countBus', counts.bus);
            updateEl('countTruck', counts.truck);
            updateEl('countBike', counts.bike);
            updateEl('countAuto', counts.auto);

            // Compute Weighted Congestion Index:
            // Truck=3.5, Bus=3.0, Car=1.0, Auto=0.8, Bike=0.5
            let weightedCongestion = (
                counts.truck * 3.5 +
                counts.bus * 3.0 +
                counts.car * 1.0 +
                counts.auto * 0.8 +
                counts.bike * 0.5
            );

            // If backend provided weighted_congestion in intersections, we can average it or use max
            if (Array.isArray(intersections) && intersections.length > 0 && intersections[0].weighted_congestion !== undefined) {
                const sum = intersections.reduce((acc, curr) => acc + (curr.weighted_congestion || 0), 0);
                weightedCongestion = Math.round((sum / intersections.length) * 10) / 10;
            } else {
                weightedCongestion = Math.round(weightedCongestion * 10) / 10;
            }

            const elWeightVal = document.getElementById('weightedCongestionVal');
            if (elWeightVal) elWeightVal.textContent = weightedCongestion.toFixed(1);

            // Update Congestion Bar & Level Label
            let level = 'low';
            let pillClass = 'pill-low';
            let fillPct = Math.min(100, Math.max(5, (weightedCongestion / 90) * 100));

            if (weightedCongestion >= 75) {
                level = 'Critical';
                pillClass = 'pill-critical';
            } else if (weightedCongestion >= 55) {
                level = 'High';
                pillClass = 'pill-high';
            } else if (weightedCongestion >= 25) {
                level = 'Medium';
                pillClass = 'pill-medium';
            } else {
                level = 'Low';
                pillClass = 'pill-low';
            }

            const elCongestionLabel = document.getElementById('congestionLabel');
            if (elCongestionLabel) {
                elCongestionLabel.textContent = level;
                elCongestionLabel.className = `status-pill ${pillClass}`;
            }

            const elBarFill = document.getElementById('congestionBarFill');
            if (elBarFill) {
                elBarFill.style.width = `${fillPct}%`;
            }
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
    window.VehicleLayer = VehicleLayer;
})();
