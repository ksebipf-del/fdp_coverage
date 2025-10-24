from qgis.PyQt import QtWidgets, uic
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox
from qgis.core import (
    QgsVectorLayer, 
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransformContext,
    QgsVectorFileWriter,
    QgsProcessingFeatureSourceDefinition,
    QgsWkbTypes
)
import processing
import os

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'fdp_coverage_dockwidget_base.ui'))

class FdpCoverageToolDockWidget(QtWidgets.QDockWidget, FORM_CLASS):

    closingPlugin = pyqtSignal()

    def __init__(self, parent=None):
        super(FdpCoverageToolDockWidget, self).__init__(parent)
        self.setupUi(self)

        self.btnBrowseFdp.clicked.connect(self.pick_fdp_csv)
        self.btnBrowseWorldpop.clicked.connect(self.pick_worldpop)
        self.btnBrowseBoundary.clicked.connect(self.pick_boundary)
        self.btnLoadLayers.clicked.connect(self.load_layers_clicked)
        self.btnCreateBuffer.clicked.connect(self.create_buffer_layer)
        self.btnAnalyzeUncovered.clicked.connect(self.analyze_uncovered_area)
        self.btnCreateHeatmap.clicked.connect(self.create_heatmap)
        self.btnOptimizePoints.clicked.connect(self.optimize_points)
        if hasattr(self, 'btnAddNewPoints'):
            self.btnAddNewPoints.clicked.connect(self.add_points_to_threshold)

        # SpinBox für Threshold konfigurieren (0–100 %, Schrittweite 1)
        if hasattr(self, 'sbTreshold'):
            self.sbTreshold.setRange(0, 100)
            self.sbTreshold.setSingleStep(1)
            self.sbTreshold.setSuffix(" %")

        # Checkbox für maximale Coverage hinzufügen
        if hasattr(self, 'checkMaxCoverage'):
            self.checkMaxCoverage.setChecked(False)  # Standardmäßig deaktiviert

    def closeEvent(self, event):
        self.closingPlugin.emit()
        event.accept()

    #  Browse function: FDP CSV
    def pick_fdp_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FDP CSV file",
            "",
            "CSV files (*.csv);;All files (*.*)"
        )
        if path:
            self.leFdpCsv.setText(path)

    #  Browse function: WorldPop CSV
    def pick_worldpop(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select WorldPop CSV file",
            "",
            "CSV files (*.csv);;All files (*.*)"
        )
        if path:
            self.leWorldpop.setText(path)

    #  Browse function: Boundary shapefile
    def pick_boundary(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select boundary shapefile",
            "",
            "Shapefile (*.shp);;All files (*.*)"
        )
        if path:
            self.leBoundary.setText(path)

    #  Load all layers
    def load_layers_clicked(self):
        fdp_csv = self.leFdpCsv.text().strip()
        worldpop = self.leWorldpop.text().strip()
        boundary = self.leBoundary.text().strip()

        missing = []
        if not os.path.isfile(fdp_csv):    
            missing.append("FDP CSV")
        if not os.path.isfile(worldpop):   
            missing.append("WorldPop CSV")
        if not os.path.isfile(boundary):   
            missing.append("Boundary shapefile")

        if missing:
            QMessageBox.warning(self, "Missing input",
                "Please check the following files:\n- " + "\n- ".join(missing))
            return

        # Load FDP CSV as point layer
        fdp_uri = self._build_csv_point_uri(fdp_csv)
        if not fdp_uri:
            QMessageBox.critical(self, "CSV Error",
                "Could not detect coordinate columns in FDP CSV.")
            return

        fdp_layer = QgsVectorLayer(fdp_uri, "FDP_Points", "delimitedtext")
        if not fdp_layer.isValid():
            QMessageBox.critical(self, "Error", "Could not load FDP CSV as layer.")
            return

        # Load WorldPop as CSV point layer
        wp_uri = self._build_csv_point_uri(worldpop)
        if not wp_uri:
            QMessageBox.critical(self, "CSV Error",
                "Could not detect coordinate columns in WorldPop CSV.")
            return

        wp_layer = QgsVectorLayer(wp_uri, "WorldPop", "delimitedtext")
        if not wp_layer.isValid():
            QMessageBox.critical(self, "Error", 
                "Could not load WorldPop CSV as layer.")
            return

        # Load boundary as vector layer
        bnd_layer = QgsVectorLayer(boundary, "Planning_Area", "ogr")
        if not bnd_layer.isValid():
            QMessageBox.critical(self, "Error", "Invalid boundary shapefile.")
            return

        # Add layers to project
        QgsProject.instance().addMapLayer(fdp_layer)
        QgsProject.instance().addMapLayer(wp_layer)
        QgsProject.instance().addMapLayer(bnd_layer)

        QMessageBox.information(self, "Success", "All layers have been loaded.")

    #  Build CSV URI dynamically
    def _build_csv_point_uri(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return None

        header_original = [h.strip() for h in header]
        header_lower = [h.lower() for h in header_original]

        # Map lowercase header to original case
        header_map = dict(zip(header_lower, header_original))

        # Known coordinate field name pairs (y, x)
        candidates = [
            ("latitude", "longitude"),
            ("lat", "long"),
            ("lat", "lon"),
            ("y", "x"),
        ]

        for y_key, x_key in candidates:
            if y_key in header_map and x_key in header_map:
                y_field = header_map[y_key]
                x_field = header_map[x_key]
                return (
                    f'file:///{path}'
                    f'?type=csv&delimiter=,&xField={x_field}&yField={y_field}&crs=EPSG:4326'
                )

        # No matching coordinates found
        print("No valid coordinate columns found in CSV.")
        print("Header was:", header_original)
        return None

    #  Create buffer layer
    def create_buffer_layer(self):
        # Find FDP Layer by name
        fdp_layer = None
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name() == "FDP_Points":
                fdp_layer = layer
                break

        if not fdp_layer:
            QMessageBox.critical(self, "Error", "FDP Points layer not found. Please load it first.")
            return

        # Buffer-Radius (km-Durchmesser → m-Radius)
        buffer_distance = (self.sbBufferDistance.value() / 2) * 1000

        # Neu: UTM-CRS automatisch aus Planning_Area
        planning_layer = None
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name().startswith("Planning_Area"):
                planning_layer = layer
                break
        if not planning_layer:
            QMessageBox.critical(self, "Error", "Planning Area layer not found.")
            return
        utm_crs = self._auto_utm_crs(planning_layer)

        try:
            reprojected = processing.run("native:reprojectlayer", {
                'INPUT': fdp_layer,
                'TARGET_CRS': utm_crs,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Reprojection failed:\n{str(e)}")
            return

        try:
            # Step 2: Create buffer around reprojected points
            buffer_result = processing.run("native:buffer", {
                'INPUT': reprojected,
                'DISTANCE': buffer_distance,
                'SEGMENTS': 25,
                'END_CAP_STYLE': 0,  # Round
                'JOIN_STYLE': 0,     # Round
                'MITER_LIMIT': 2,
                'DISSOLVE': False,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })
            buffer_layer = buffer_result['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Buffer creation failed:\n{str(e)}")
            return

        # Style buffer: light gray transparent fill with black outline
        from qgis.core import QgsFillSymbol, QgsSingleSymbolRenderer
        from qgis.PyQt.QtGui import QColor

        symbol = QgsFillSymbol.createSimple({
            'color': '200,200,200,60',     # light gray with transparency
            'outline_color': '0,0,0,255',  # black outline
            'outline_width': '0.6'         # 0.6 mm
        })
        buffer_layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        # Step 3: Add buffer layer to map
        buffer_layer.setName(f"FDP_Buffer_{round(buffer_distance / 1000, 2)}km_radius")
        QgsProject.instance().addMapLayer(buffer_layer)

        QMessageBox.information(self, "Success", f"Buffer created with {round(buffer_distance / 1000, 2)} km radius.")

    #  Analyze uncovered area
    def analyze_uncovered_area(self):
        from qgis.core import QgsProject, QgsVectorLayer
        import processing

        # Layers holen
        worldpop_layer = None
        planning_layer = None
        fdp_layer = None  # Nutze FDP_Points (oder Optimized, falls vorhanden)

        for layer in QgsProject.instance().mapLayers().values():
            if layer.name().startswith("WorldPop"):
                worldpop_layer = layer
            elif layer.name().startswith("Planning_Area"):
                planning_layer = layer

        # Priorisiere Optimized_FDP_Points, falls vorhanden
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name() == "Optimized_FDP_Points":
                fdp_layer = layer
                break
        if fdp_layer is None:
            for layer in QgsProject.instance().mapLayers().values():
                if layer.name() == "FDP_Points":
                    fdp_layer = layer
                    break

        if not all([worldpop_layer, planning_layer, fdp_layer]):
            QMessageBox.critical(self, "Error", "WorldPop, Planning Area and FDP Points must be loaded.")
            return

        # Buffer-Radius (Meter) direkt aus sbBufferDistance
        buffer_distance = self._buffer_distance_m()
        if buffer_distance <= 0:
            QMessageBox.critical(self, "Error", "Invalid buffer distance. Please set sbBufferDistance > 0.")
            return

        # UTM-CRS aus Planning_Area
        utm_crs = self._auto_utm_crs(planning_layer)

        # 1. Clip WorldPop with Planning Area
        try:
            clipped_pop = processing.run("native:clip", {
                'INPUT': worldpop_layer,
                'OVERLAY': planning_layer,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Clipping WorldPop failed:\n{str(e)}")
            return

        # 2. Reprojecte FDP nach UTM und erstelle Buffer
        try:
            fdp_utm = processing.run("native:reprojectlayer", {
                'INPUT': fdp_layer,
                'TARGET_CRS': utm_crs,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
            buffer_layer = processing.run("native:buffer", {
                'INPUT': fdp_utm,
                'DISTANCE': buffer_distance,
                'SEGMENTS': 25,
                'END_CAP_STYLE': 0,
                'JOIN_STYLE': 0,
                'MITER_LIMIT': 2,
                'DISSOLVE': True,  # Union aller Buffer
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Buffer creation failed:\n{str(e)}")
            return

        # 3. Erstelle Difference-Maske (nicht abgedeckte Gebiete = Planning - Buffer)
        try:
            uncovered_geom = processing.run("native:difference", {
                'INPUT': planning_layer,
                'OVERLAY': buffer_layer,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Difference calculation failed:\n{str(e)}")
            return

        # 4. Clip WorldPop mit uncovered_geom → Population außerhalb der Buffer
        try:
            uncovered_pop = processing.run("native:clip", {
                'INPUT': clipped_pop,
                'OVERLAY': uncovered_geom,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Clipping uncovered areas failed:\n{str(e)}")
            return

        # 5. Bevölkerung berechnen
        total = sum([f['Z'] for f in clipped_pop.getFeatures() if f['Z'] is not None])
        uncovered = sum([f['Z'] for f in uncovered_pop.getFeatures() if f['Z'] is not None])
        covered = total - uncovered
        percent_covered = round((covered / total) * 100, 1) if total > 0 else 0

        # Update widgets (Covered)
        self.leTotalPop.setText(str(int(total)))
        self.leCoveredPop.setText(str(int(covered)))
        self.lblPercentageValue.setText(f"{percent_covered}%")

        QMessageBox.information(self, "Success",
            f"Analysis complete!\n\n"
            f"Total population: {int(total):,}\n"
            f"Covered by {round(buffer_distance/1000, 2)} km radius: {int(covered):,} ({percent_covered}%)")

    def create_heatmap(self):
        from qgis.core import (
            QgsVectorLayer, QgsProject, QgsFeature, QgsField,
            QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsSymbol
        )
        from qgis.PyQt.QtCore import QVariant
        from qgis.PyQt.QtGui import QColor
        import processing, math

        # WorldPop layer
        worldpop_layer = None
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name().startswith("WorldPop"):
                worldpop_layer = layer
                break
        if not worldpop_layer:
            QMessageBox.critical(self, "Error", "WorldPop layer not found.")
            return

        # Planning area
        planning_layer = None
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name().startswith("Planning_Area"):
                planning_layer = layer
                break
        if not planning_layer:
            QMessageBox.critical(self, "Error", "Planning Area not found.")
            return

        # Clip to planning area
        try:
            clipped = processing.run("native:clip", {
                'INPUT': worldpop_layer,
                'OVERLAY': planning_layer,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Clipping failed:\n{str(e)}")
            return

        # Collect Z values
        z_values = [f['Z'] for f in clipped.getFeatures() if f['Z'] is not None]
        if not z_values:
            QMessageBox.critical(self, "Error", "No Z values found in WorldPop.")
            return

        z_values.sort()

        def percentile(sorted_vals, p):
            if not sorted_vals:
                return None
            k = (len(sorted_vals) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return sorted_vals[int(k)]
            return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

        # Thresholds: p90–p95 (Blue), >= p95 (Orange)
        p90 = percentile(z_values, 0.90)
        p95 = percentile(z_values, 0.95)

        # Build filtered layer (only >= 90%)
        fields = clipped.fields()
        fields.append(QgsField('category', QVariant.String))

        out_layer = QgsVectorLayer(f"Point?crs={clipped.crs().authid()}", "Population_Hotspots", "memory")
        dp = out_layer.dataProvider()
        dp.addAttributes(fields)
        out_layer.updateFields()

        feats = []
        cnt_orange = 0  # top 5%
        cnt_blue = 0    # top 5–10%
        for feat in clipped.getFeatures():
            z = feat['Z']
            if z is None:
                continue
            if z >= p90:
                newf = QgsFeature(fields)
                newf.setGeometry(feat.geometry())
                for fld in clipped.fields():
                    newf[fld.name()] = feat[fld.name()]
                if z >= p95:
                    newf['category'] = 'top_5'       # Orange (>= 95th)
                    cnt_orange += 1
                else:
                    newf['category'] = 'top_5_10'    # Blue (90th–95th)
                    cnt_blue += 1
                feats.append(newf)

        if not feats:
            QMessageBox.information(self, "Info", "No points meet the thresholds (>= 90th percentile).")
            return

        dp.addFeatures(feats)

        # Style
        cats = []
        sym_orange = QgsSymbol.defaultSymbol(out_layer.geometryType())
        sym_orange.setColor(QColor(255, 140, 0))  # Orange
        sym_orange.setSize(2.0)
        cats.append(QgsRendererCategory('top_5', sym_orange, 'Top 5%'))

        sym_blue = QgsSymbol.defaultSymbol(out_layer.geometryType())
        sym_blue.setColor(QColor(0, 120, 255))    # Blue
        sym_blue.setSize(1.6)
        cats.append(QgsRendererCategory('top_5_10', sym_blue, 'Top 5–10%'))

        out_layer.setRenderer(QgsCategorizedSymbolRenderer('category', cats))
        QgsProject.instance().addMapLayer(out_layer)

        QMessageBox.information(self, "Success",
                                f"Hotspots created:\nOrange (top 5%): {cnt_orange}\nBlue (top 5–10%): {cnt_blue}")

    def optimize_points(self):
        """
        Re-Arrange only the closest FDP points:
        - Keep non-candidate (far apart) FDPs fixed
        - Select candidates = FDPs with overlapping buffers or smallest NN-distance
        - Greedily relocate only these candidates to maximize uncovered population coverage
        - Accept result only if coverage improves vs. baseline
        """
        from qgis.core import (
            QgsVectorLayer, QgsProject, QgsFeature, QgsField, QgsFields,
            QgsSpatialIndex, QgsPointXY, QgsGeometry, QgsCoordinateReferenceSystem,
            QgsMarkerSymbol, QgsSingleSymbolRenderer, QgsCategorizedSymbolRenderer, QgsRendererCategory
        )
        from qgis.PyQt.QtCore import QVariant
        from qgis.PyQt.QtGui import QColor
        import processing, math

        # Get layers
        worldpop_layer = None
        planning_layer = None
        fdp_layer = None

        for layer in QgsProject.instance().mapLayers().values():
            if layer.name().startswith("WorldPop"):
                worldpop_layer = layer
            elif layer.name().startswith("Planning_Area"):
                planning_layer = layer
            elif layer.name() == "FDP_Points":
                fdp_layer = layer

        if not all([worldpop_layer, planning_layer, fdp_layer]):
            QMessageBox.critical(self, "Error",
                                 "All three layers (WorldPop, Planning Area, FDP Points) must be loaded.")
            return

        # Buffer-Radius (km-Durchmesser → m-Radius)
        buffer_distance = (self.sbBufferDistance.value() / 2) * 1000

        # UTM-CRS aus Planning_Area
        utm_crs = self._auto_utm_crs(planning_layer)

        # 1) Clip WorldPop to Planning Area
        try:
            clipped_pop = processing.run("native:clip", {
                'INPUT': worldpop_layer,
                'OVERLAY': planning_layer,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"WorldPop clipping failed:\n{str(e)}")
            return

        # 2) Reproject WorldPop + FDP to UTM (meters)
        try:
            pop_utm = processing.run("native:reprojectlayer", {
                'INPUT': clipped_pop,
                'TARGET_CRS': utm_crs,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
            fdp_utm = processing.run("native:reprojectlayer", {
                'INPUT': fdp_layer,
                'TARGET_CRS': utm_crs,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Reprojection failed:\n{str(e)}")
            return

        # 3) Build WorldPop dict + spatial index
        pop_features = {}
        for feat in pop_utm.getFeatures():
            z = feat['Z'] if feat['Z'] is not None else 0
            pop_features[feat.id()] = {
                'geom': feat.geometry(),
                'point': feat.geometry().asPoint(),
                'pop': z,
                'covered': False
            }
        if not pop_features:
            QMessageBox.critical(self, "Error", "No WorldPop points found after clipping.")
            return

        # NEU: Gesamtbevölkerung berechnen
        total_pop = sum(p['pop'] for p in pop_features.values())
        if total_pop <= 0:
            QMessageBox.critical(self, "Error", "No population data in planning area.")
            return

        pop_index = QgsSpatialIndex(pop_utm.getFeatures())

        # Helper to mark coverage given a list of point buffers
        def mark_coverage(buffers, covered_dict):
            for b in buffers:
                nearby_ids = pop_index.intersects(b.boundingBox())
                for pid in nearby_ids:
                    pdata = pop_features.get(pid)
                    if pdata and not covered_dict[pid]:
                        if b.contains(pdata['geom']):
                            covered_dict[pid] = True

        # 4) Build FDP buffers and nearest-neighbor distances
        fdp_feats = list(fdp_utm.getFeatures())
        if not fdp_feats:
            QMessageBox.critical(self, "Error", "No FDP points found.")
            return

        # Prepare buffers list
        fdp_data = []
        for f in fdp_feats:
            pt = f.geometry().asPoint()
            buf = f.geometry().buffer(buffer_distance, 25)
            fdp_data.append({'feat': f, 'point': pt, 'buffer': buf})

        # Compute nearest-neighbor distance for each FDP (O(n^2) is OK for moderate n)
        def sqr_dist(a, b):
            dx = a.x() - b.x()
            dy = a.y() - b.y()
            return dx*dx + dy*dy

        for i, fi in enumerate(fdp_data):
            nn2 = float('inf')
            overlaps = False
            for j, fj in enumerate(fdp_data):
                if i == j:
                    continue
                # NN distance
                d2 = sqr_dist(fi['point'], fj['point'])
                if d2 < nn2:
                    nn2 = d2
                # Buffer overlap
                if not overlaps and fi['buffer'].intersects(fj['buffer']):
                    overlaps = True
            fi['nn_dist'] = math.sqrt(nn2) if nn2 != float('inf') else float('inf')
            fi['overlaps'] = overlaps

        # 5) Select candidates: overlapping OR nn_dist < buffer_distance
        candidates = [fi for fi in fdp_data if fi['overlaps'] or fi['nn_dist'] < buffer_distance]
        # Limit to the closest ones to avoid relocating too many
        max_ratio = 0.3  # relocate up to 30% of points
        max_candidates = max(1, int(len(fdp_data) * max_ratio))
        candidates.sort(key=lambda x: (not x['overlaps'], x['nn_dist']))  # overlaps first, then smallest NN
        candidates = candidates[:min(len(candidates), max_candidates)]

        kept = [fi for fi in fdp_data if fi not in candidates]

        # 6) Compute baseline coverage (all FDP as they are)
        baseline_covered = {pid: False for pid in pop_features.keys()}
        mark_coverage([fi['buffer'] for fi in fdp_data], baseline_covered)
        baseline_cov_pop = sum(pop_features[pid]['pop'] for pid, c in baseline_covered.items() if c)
        baseline_pct = round((baseline_cov_pop / total_pop) * 100, 1) if total_pop > 0 else 0.0

        # 7) Start relocation scenario: keep 'kept' fixed, relocate only 'candidates'
        # Reset coverage and mark coverage from kept first
        covered = {pid: False for pid in pop_features.keys()}
        mark_coverage([fi['buffer'] for fi in kept], covered)

        # Pre-build fixed buffers for overlap check
        fixed_buffers = [fi['buffer'] for fi in kept]

        # Greedy placement of exactly len(candidates) new positions
        relocated_points = []
        for it in range(len(candidates)):
            best_point = None
            best_gain = 0

            # Try all WorldPop points as potential centers
            for pid, pdata in pop_features.items():
                # Skip if this location would overlap fixed or already placed relocated
                candidate_buffer = QgsGeometry.fromPointXY(QgsPointXY(pdata['point'])).buffer(buffer_distance, 25)

                overlaps_fixed = any(candidate_buffer.intersects(b) for b in fixed_buffers)
                if overlaps_fixed:
                    continue
                overlaps_relocated = False
                for rp in relocated_points:
                    rp_buf = QgsGeometry.fromPointXY(QgsPointXY(rp)).buffer(buffer_distance, 25)
                    if candidate_buffer.intersects(rp_buf):
                        overlaps_relocated = True
                        break
                if overlaps_relocated:
                    continue

                # Count uncovered population within candidate buffer
                gain = 0
                nearby_ids = pop_index.intersects(candidate_buffer.boundingBox())
                for nid in nearby_ids:
                    if not covered.get(nid, False):
                        if candidate_buffer.contains(pop_features[nid]['geom']):
                            gain += pop_features[nid]['pop']

                if gain > best_gain:
                    best_gain = gain
                    best_point = pdata['point']

            # Fallback: if no non-overlapping spot gives gain, relax overlap constraint
            if best_point is None:
                for pid, pdata in pop_features.items():
                    candidate_buffer = QgsGeometry.fromPointXY(QgsPointXY(pdata['point'])).buffer(buffer_distance, 25)
                    gain = 0
                    nearby_ids = pop_index.intersects(candidate_buffer.boundingBox())
                    for nid in nearby_ids:
                        if not covered.get(nid, False):
                            if candidate_buffer.contains(pop_features[nid]['geom']):
                                gain += pop_features[nid]['pop']
                    if gain > best_gain:
                        best_gain = gain
                        best_point = pdata['point']

            if best_point is None or best_gain <= 0:
                # No useful relocation for remaining candidates
                break

            # Accept this placement
            relocated_points.append(best_point)
            # Mark covered by this new buffer
            new_buf = QgsGeometry.fromPointXY(QgsPointXY(best_point)).buffer(buffer_distance, 25)
            nearby_ids = pop_index.intersects(new_buf.boundingBox())
            for nid in nearby_ids:
                if not covered.get(nid, False):
                    if new_buf.contains(pop_features[nid]['geom']):
                        covered[nid] = True

        # 8) Compare coverage vs. baseline
        final_cov_pop = sum(pop_features[pid]['pop'] for pid, c in covered.items() if c)
        final_pct = round((final_cov_pop / total_pop) * 100, 1) if total_pop > 0 else 0.0
        uncovered_pop = total_pop - final_cov_pop
        uncovered_pct = round((uncovered_pop / total_pop) * 100, 1) if total_pop > 0 else 0.0

        # Update UI (uncovered)
        self.leTotalPop_2.setText(str(int(total_pop)))
        self.leCoveredPop_opt.setText(str(int(final_cov_pop)))  # Update für abgedeckte Bevölkerung
        self.lblPercentageValue_opt.setText(f"{final_pct}%")  # Update für abgedeckte Prozentzahl

        # NEU: Coverage (%) nach der Optimierung in die SpinBox schreiben
        if hasattr(self, 'sbTreshold'):
            # Wenn keine Verbesserung, zeigen wir die Baseline-Coverage
            value_to_show = final_pct
            if 'baseline_pct' in locals() and final_cov_pop <= baseline_cov_pop:
                value_to_show = baseline_pct
            self.sbTreshold.setValue(int(round(value_to_show)))

        if final_cov_pop <= baseline_cov_pop:
            QMessageBox.information(self, "No Improvement",
                                    f"No coverage improvement by relocating closest points.\n\n"
                                    f"Baseline: {baseline_pct}% (covered {int(baseline_cov_pop):,})\n"
                                    f"Attempt:  {final_pct}% (covered {int(final_cov_pop):,})")
            return

        # 9) Build result layer: kept (fixed) + relocated (new positions), same total count as original
        fields = QgsFields()
        fields.append(QgsField('id', QVariant.Int))
        fields.append(QgsField('type', QVariant.String))  # 'fixed' or 'relocated'

        opt_layer = QgsVectorLayer("Point?crs=EPSG:32736", "Optimized_FDP_Points_UTM", "memory")
        dp = opt_layer.dataProvider()
        dp.addAttributes(fields)
        opt_layer.updateFields()

        new_feats = []
        fid = 1

        # Fixed points remain at original positions
        for k in kept:
            f = QgsFeature(opt_layer.fields())
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(k['point'])))
            f['id'] = fid
            f['type'] = 'fixed'
            new_feats.append(f)
            fid += 1

        # Relocated points at new positions
        for rp in relocated_points:
            f = QgsFeature(opt_layer.fields())
            f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(rp)))
            f['id'] = fid
            f['type'] = 'relocated'
            new_feats.append(f)
            fid += 1

        dp.addFeatures(new_feats)

        # Reproject back to WGS84 and style
        try:
            opt_layer_wgs84 = processing.run("native:reprojectlayer", {
                'INPUT': opt_layer,
                'TARGET_CRS': QgsCoordinateReferenceSystem("EPSG:4326"),
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
            opt_layer_wgs84.setName("Optimized_FDP_Points")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Back-projection failed:\n{str(e)}")
            return

        cats = []
        sym_fixed = QgsMarkerSymbol.createSimple({
            'name': 'circle', 'color': '0,120,255,255', 'size': '2.8',
            'outline_color': '0,0,0,255', 'outline_width': '0.4'
        })
        cats.append(QgsRendererCategory('fixed', sym_fixed, 'Fixed (kept)'))

        sym_relocated = QgsMarkerSymbol.createSimple({
            'name': 'circle', 'color': '0,200,0,255', 'size': '3.4',
            'outline_color': '0,0,0,255', 'outline_width': '0.5'
        })
        cats.append(QgsRendererCategory('relocated', sym_relocated, 'Relocated (optimized)'))

        opt_layer_wgs84.setRenderer(QgsCategorizedSymbolRenderer('type', cats))
        QgsProject.instance().addMapLayer(opt_layer_wgs84)

        QMessageBox.information(self, "Optimization Complete",
                                f"Relocated {len(relocated_points)} of {len(fdp_feats)} FDP points "
                                f"(candidates: {len(candidates)}).\n\n"
                                f"Baseline coverage: {baseline_pct}% ({int(baseline_cov_pop):,})\n"
                                f"New coverage:      {final_pct}% ({int(final_cov_pop):,})")

    def add_points_to_threshold(self):
        """
        Fügt neue FDP-Punkte greedy hinzu, bis der per SpinBox sbTreshold gesetzte
        Abdeckungs-Schwellenwert (in %) erreicht ist.
        Falls checkMaxCoverage aktiviert ist, wird sbTreshold ignoriert und maximale Coverage angestrebt.
        Nutzt "Optimized_FDP_Points" falls vorhanden, sonst "FDP_Points".
        Bezieht bereits existierende "New_FDP_Points" in Coverage und Abstand ein.
        """
        from qgis.core import (
            QgsVectorLayer, QgsProject, QgsFeature, QgsField, QgsFields,
            QgsSpatialIndex, QgsPointXY, QgsGeometry, QgsCoordinateReferenceSystem,
            QgsMarkerSymbol, QgsCategorizedSymbolRenderer, QgsRendererCategory,
            QgsCoordinateTransform
        )
        from qgis.PyQt.QtCore import QVariant
        import processing

        # Layers holen
        worldpop_layer = None
        planning_layer = None
        fdp_layer = None  # Optimized oder Original
        new_points_layer = None  # bereits vorhandene neue Punkte (WGS84)

        for layer in QgsProject.instance().mapLayers().values():
            if layer.name().startswith("WorldPop"):
                worldpop_layer = layer
            elif layer.name().startswith("Planning_Area"):
                planning_layer = layer
            elif layer.name() == "New_FDP_Points":
                new_points_layer = layer

        # Priorisiere Optimized
        for layer in QgsProject.instance().mapLayers().values():
            if layer.name() == "Optimized_FDP_Points":
                fdp_layer = layer
                break
        if fdp_layer is None:
            for layer in QgsProject.instance().mapLayers().values():
                if layer.name() == "FDP_Points":
                    fdp_layer = layer
                    break

        if not all([worldpop_layer, planning_layer, fdp_layer]):
            QMessageBox.critical(self, "Error", "WorldPop, Planning Area and FDP (or Optimized) layer must be loaded.")
            return

        # Prüfe, ob maximale Coverage angestrebt wird
        maximize_coverage = False
        if hasattr(self, 'checkMaxCoverage'):
            maximize_coverage = self.checkMaxCoverage.isChecked()

        # Threshold (0–100) aus SpinBox (wird bei maximize_coverage ignoriert)
        target_pct = 100.0  # Default: maximale Coverage
        if not maximize_coverage:
            if not hasattr(self, 'sbTreshold'):
                QMessageBox.critical(self, "Error", "SpinBox 'sbTreshold' not found in UI.")
                return
            target_pct = max(0, min(100, int(self.sbTreshold.value())))

        # Buffer-Radius (Meter) direkt aus sbBufferDistance
        buffer_distance = self._buffer_distance_m()
        if buffer_distance <= 0:
            QMessageBox.critical(self, "Error", "Invalid buffer distance. Please set sbBufferDistance > 0.")
            return

        # WorldPop auf Planning Area clippen
        try:
            clipped_pop = processing.run("native:clip", {
                'INPUT': worldpop_layer,
                'OVERLAY': planning_layer,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"WorldPop clipping failed:\n{str(e)}")
            return

        # UTM-CRS aus Planning_Area
        utm_crs = self._auto_utm_crs(planning_layer)

        # Reproject nach UTM (Meter)
        try:
            pop_utm = processing.run("native:reprojectlayer", {
                'INPUT': clipped_pop,
                'TARGET_CRS': utm_crs,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
            fdp_utm = processing.run("native:reprojectlayer", {
                'INPUT': fdp_layer,
                'TARGET_CRS': utm_crs,
                'OUTPUT': 'TEMPORARY_OUTPUT'
            })['OUTPUT']
            new_points_utm = None
            if new_points_layer is not None:
                new_points_utm = processing.run("native:reprojectlayer", {
                    'INPUT': new_points_layer,
                    'TARGET_CRS': utm_crs,
                    'OUTPUT': 'TEMPORARY_OUTPUT'
                })['OUTPUT']
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Reprojection failed:\n{str(e)}")
            return

        # WorldPop-Features + Index
        pop_features = {}
        for feat in pop_utm.getFeatures():
            z = feat['Z'] if feat['Z'] is not None else 0
            pop_features[feat.id()] = {
                'geom': feat.geometry(),
                'point': feat.geometry().asPoint(),
                'pop': z
            }
        if not pop_features:
            QMessageBox.critical(self, "Error", "No WorldPop points found after clipping.")
            return
        total_pop = sum(p['pop'] for p in pop_features.values())
        if total_pop <= 0:
            QMessageBox.critical(self, "Error", "No population data in planning area.")
            return
        pop_index = QgsSpatialIndex(pop_utm.getFeatures())

        # Helper: IDs, die ein Buffer abdeckt
        def ids_covered_by_buffer(buffer_geom):
            ids = set()
            for pid in pop_index.intersects(buffer_geom.boundingBox()):
                pf = pop_features.get(pid)
                if pf and buffer_geom.contains(pf['geom']):
                    ids.add(pid)
            return ids

        # FDP-Buffer (Optimized/Original) + bereits vorhandene New_Points
        existing_buffers = []
        for f in fdp_utm.getFeatures():
            existing_buffers.append(f.geometry().buffer(buffer_distance, 25))
        if new_points_utm is not None:
            for nf in new_points_utm.getFeatures():
                existing_buffers.append(nf.geometry().buffer(buffer_distance, 25))

        # Start-Coverage (Union der IDs aus allen vorhandenen Buffern)
        covered_ids = set()
        for b in existing_buffers:
            covered_ids |= ids_covered_by_buffer(b)
        covered_pop = sum(pop_features[i]['pop'] for i in covered_ids)
        covered_pct = (covered_pop / total_pop) * 100 if total_pop > 0 else 0.0

        # Ziel bereits erreicht? (nur prüfen wenn nicht maximize_coverage)
        if not maximize_coverage and covered_pct >= target_pct:
            self.leTotalPop_3.setText(str(int(total_pop)))
            if hasattr(self, 'leCoveredPop_add'):
                self.leCoveredPop_add.setText(str(int(covered_pop)))
            if hasattr(self, 'lblPercentageValue_add'):
                self.lblPercentageValue_add.setText(f"{round(covered_pct, 1)}%")
            if hasattr(self, 'sbTreshold'):
                self.sbTreshold.setValue(int(round(covered_pct)))
            QMessageBox.information(self, "Threshold reached",
                                    f"Target coverage {target_pct}% already met.\n"
                                    f"Covered: {int(covered_pop):,} ({round(covered_pct, 1)}%)")
            return

        # Greedy: neue Punkte hinzufügen bis Threshold erreicht oder kein Zugewinn mehr
        # Bei maximize_coverage: bis kein Zugewinn mehr möglich
        new_points = []
        new_buffers = []  # Buffers der neu hinzugefügten Punkte, für Abstands-/Overlap-Check
        
        while True:
            # Abbruchbedingung: Threshold erreicht (außer bei maximize_coverage)
            if not maximize_coverage and covered_pct >= target_pct:
                break
            
            best_point = None
            best_ids = set()
            best_gain = 0

            # Kandidaten = alle WorldPop-Punkte
            for pid, pdata in pop_features.items():
                pt = pdata['point']
                pt_geom = QgsGeometry.fromPointXY(QgsPointXY(pt))

                # 1) Punkt selbst darf nicht innerhalb bestehender oder neuer Buffer liegen
                if any(b.contains(pt_geom) for b in existing_buffers):
                    continue
                if any(b.contains(pt_geom) for b in new_buffers):
                    continue

                cand_buf = QgsGeometry.fromPointXY(QgsPointXY(pt)).buffer(buffer_distance, 25)

                # 2) Der Buffer des neuen Punkts darf bestehende oder neue Buffer nicht schneiden
                if any(cand_buf.intersects(b) for b in existing_buffers):
                    continue
                if any(cand_buf.intersects(b) for b in new_buffers):
                    continue

                # Zugewinn = noch nicht abgedeckte IDs in diesem Buffer
                gain_ids = ids_covered_by_buffer(cand_buf) - covered_ids
                if not gain_ids:
                    continue
                gain = sum(pop_features[i]['pop'] for i in gain_ids)
                if gain > best_gain:
                    best_gain = gain
                    best_point = pt
                    best_ids = gain_ids

            if best_point is None or best_gain <= 0:
                break  # keine Verbesserung möglich (unter Abstandsbedingung)

            # Besten Punkt übernehmen
            new_points.append(best_point)
            nb = QgsGeometry.fromPointXY(QgsPointXY(best_point)).buffer(buffer_distance, 25)
            new_buffers.append(nb)
            covered_ids |= best_ids
            covered_pop += best_gain
            covered_pct = (covered_pop / total_pop) * 100

        # Finale Ausgabe in UI (covered) für Kombination: Optimized + vorhandene New + neue New
        final_pct = round(covered_pct, 1)
        self.leTotalPop_3.setText(str(int(total_pop)))
        self.leCoveredPop_add.setText(str(int(covered_pop)))
        self.lblPercentageValue_add.setText(f"{final_pct}%")
        if hasattr(self, 'sbTreshold'):
            self.sbTreshold.setValue(int(round(final_pct)))

        # Falls keine Punkte hinzukamen
        if not new_points:
            mode_text = "maximize coverage" if maximize_coverage else f"reach {target_pct}%"
            QMessageBox.information(self, "No improvement possible",
                                    f"Could not add points to {mode_text}.\n"
                                    f"Covered: {int(covered_pop):,} ({final_pct}%)")
            return

        # Neue Punkte in bestehenden New_FDP_Points-Layer anhängen oder neuen erstellen
        try:
            if new_points_layer is not None:
                # Append im WGS84-Layer
                from qgis.core import QgsCoordinateTransformContext
                crs_src = utm_crs
                crs_dst = QgsCoordinateReferenceSystem("EPSG:4326")
                xform = QgsCoordinateTransform(crs_src, crs_dst, QgsProject.instance().transformContext())

                prov = new_points_layer.dataProvider()
                fields = new_points_layer.fields()
                next_id = (max([f['id'] for f in new_points_layer.getFeatures()] or [0]) + 1)

                feats = []
                for i, pt in enumerate(new_points, start=next_id):
                    wgs_pt = xform.transform(QgsPointXY(pt))
                    f = QgsFeature(fields)
                    f.setGeometry(QgsGeometry.fromPointXY(wgs_pt))
                    f['id'] = i
                    f['type'] = 'new'
                    feats.append(f)
                prov.addFeatures(feats)
                new_points_layer.updateExtents()
            else:
                # Neu erstellen (wie bisher)
                fields = QgsFields()
                fields.append(QgsField('id', QVariant.Int))
                fields.append(QgsField('type', QVariant.String))

                new_layer_utm = QgsVectorLayer(f"Point?crs={utm_crs.authid()}", "New_FDP_Points_UTM", "memory")
                dp = new_layer_utm.dataProvider()
                dp.addAttributes(fields)
                new_layer_utm.updateFields()

                feats = []
                for i, pt in enumerate(new_points, start=1):
                    f = QgsFeature(new_layer_utm.fields())
                    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(pt)))
                    f['id'] = i
                    f['type'] = 'new'
                    feats.append(f)
                dp.addFeatures(feats)

                new_layer = processing.run("native:reprojectlayer", {
                    'INPUT': new_layer_utm,
                    'TARGET_CRS': QgsCoordinateReferenceSystem("EPSG:4326"),
                    'OUTPUT': 'TEMPORARY_OUTPUT'
                })['OUTPUT']
                new_layer.setName("New_FDP_Points")

                # Style: neue Punkte grün (Stern)
                sym_new = QgsMarkerSymbol.createSimple({
                    'name': 'star', 'color': '0,200,0,255', 'size': '4',
                    'outline_color': '0,0,0,255', 'outline_width': '0.5'
                })
                renderer = QgsCategorizedSymbolRenderer('type', [
                    QgsRendererCategory('new', sym_new, 'New FDP points')
                ])
                new_layer.setRenderer(renderer)
                QgsProject.instance().addMapLayer(new_layer)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write New_FDP_Points:\n{str(e)}")
            return

        mode_text = "maximize coverage" if maximize_coverage else f"reach target {target_pct}%"
        QMessageBox.information(self, "Points added",
                                f"Added {len(new_points)} new FDP points to {mode_text}.\n"
                                f"Covered: {int(covered_pop):,} ({final_pct}%)")

    # Helper: aktuelle Buffer-Distanz (Meter) aus sbBufferDistance
    def _buffer_distance_m(self) -> float:
        """
        Liest den UI-Wert aus sbBufferDistance (Durchmesser in km),
        wandelt zu Radius in Metern um: (wert / 2) * 1000.
        """
        if not hasattr(self, 'sbBufferDistance'):
            QMessageBox.critical(self, "Error", "UI control 'sbBufferDistance' not found.")
            return 0.0
        try:
            val = float(self.sbBufferDistance.value())
        except Exception:
            QMessageBox.critical(self, "Error", "Invalid buffer distance value.")
            return 0.0
        if val <= 0:
            return 0.0
        return (val / 2.0) * 1000.0

    def _auto_utm_crs(self, planning_layer) -> QgsCoordinateReferenceSystem:
        """
        Bestimmt ein passendes Meter-CRS:
        - UTM-Zone aus dem Mittelpunkt der Planning_Area (in WGS84)
        - Fallbacks: Polar-CRS bei |lat| > 84°
        """
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform
        # Mittelpunkt der Planning_Area
        crs_planning = planning_layer.crs()
        crs_wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        tr = QgsCoordinateTransform(crs_planning, crs_wgs84, QgsProject.instance().transformContext())
        centroid_wgs = tr.transform(planning_layer.extent().center())
        lat = centroid_wgs.y()
        lon = centroid_wgs.x()

        # Polar-Fallback
        if lat >= 84:
            return QgsCoordinateReferenceSystem("EPSG:3413")  # NSIDC Polar Stereographic North
        if lat <= -80:
            return QgsCoordinateReferenceSystem("EPSG:3031")  # Antarctic Polar Stereographic

        # UTM-Zone bestimmen
        zone = int((lon + 180) // 6) + 1
        zone = max(1, min(60, zone))
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return QgsCoordinateReferenceSystem(f"EPSG:{epsg}")

