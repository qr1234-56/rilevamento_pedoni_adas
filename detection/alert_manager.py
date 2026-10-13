"""
Gestione dei livelli di allerta VERDE / GIALLO / ROSSO.

Il livello assegnato a ciascun pedone dipende dalla distanza stimata in
metri e dal tempo residuo alla collisione (time-to-collision).

Le soglie predefinite derivano dallo spazio di arresto in ambito urbano.
Un veicolo a 50 km/h percorre circa 14 m nel solo tempo di reazione del
conducente e necessita complessivamente di circa 27 m per fermarsi; a
30 km/h lo spazio complessivo scende attorno ai 13 m. Di conseguenza:

    oltre 25 m      -> VERDE   : il veicolo ha spazio per arrestarsi
    fra 10 e 25 m   -> GIALLO  : l'arresto richiede una frenata decisa
    entro 10 m      -> ROSSO   : a velocità urbana l'arresto non è garantito

I valori sono parametri del costruttore e vanno ritarati sullo scenario
effettivamente considerato.

Alla distanza si affianca un criterio di corridoio: conta se il pedone si
trova, o sta per trovarsi, sulla traiettoria del veicolo. Lo scostamento
laterale viene stimato in metri e confrontato con la semilarghezza del
corridoio di marcia; chi resta fuori viene declassato di un livello. La
previsione tiene conto della velocita' laterale, in modo da segnalare in
anticipo chi sta attraversando e non solo chi e' gia' davanti.
"""


class AlertManager:
    """
    Determina il livello di allerta associato a ciascun pedone tracciato.

    La classe mantiene uno stato per ogni identificativo di traccia, in modo
    da poter applicare isteresi e conferma temporale ed evitare che il
    livello oscilli da un frame all'altro.
    """

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

    _SEVERITY = {GREEN: 0, YELLOW: 1, RED: 2}

    def __init__(
        self,
        red_distance_m=10.0,
        yellow_distance_m=25.0,
        hysteresis_m=2.5,
        red_ttc_s=2.0,
        yellow_ttc_s=4.0,
        corridor_half_width_m=1.5,
        corridor_hysteresis_m=0.5,
        lateral_horizon_s=3.0,
        min_confidence_yellow=0.50,
        min_confidence_red=0.35,
        frames_to_escalate=1,
        frames_to_escalate_ttc=2,
        frames_to_relax=5,
        max_truncated_hold=15,
    ):
        self.red_distance_m = red_distance_m

        self.yellow_distance_m = yellow_distance_m

        # Isteresi: evita il lampeggio quando la distanza oscilla sulla soglia.
        self.hysteresis_m = hysteresis_m

        # TTC critico: vale anche a distanza maggiore.
        self.red_ttc_s = red_ttc_s

        # TTC che giustifica il GIALLO.
        self.yellow_ttc_s = yellow_ttc_s

        # Mezza larghezza del veicolo piu' un margine.
        self.corridor_half_width_m = corridor_half_width_m

        # Allargamento del corridoio per le tracce gia' segnalate.
        self.corridor_hysteresis_m = corridor_hysteresis_m

        # Orizzonte massimo della previsione laterale.
        self.lateral_horizon_s = lateral_horizon_s

        self.min_confidence_yellow = min_confidence_yellow

        # Più bassa di quella del GIALLO: un pedone mancato pesa più di un falso allarme.
        self.min_confidence_red = min_confidence_red

        # Il valore predefinito è 1: sulla distanza l'innalzamento è immediato.
        self.frames_to_escalate = max(1, int(frames_to_escalate))

        # Il tempo al contatto è la derivata di una distanza rumorosa: un
        # singolo valore anomalo non deve far scattare l'allerta.
        self.frames_to_escalate_ttc = max(1, int(frames_to_escalate_ttc))

        # Più di un frame per abbassare il livello: filtra il rumore.
        self.frames_to_relax = max(1, int(frames_to_relax))

        # Limite al blocco da troncamento: senza, resterebbe ROSSO per sempre.
        self.max_truncated_hold = max(0, int(max_truncated_hold))

        self._states = {}


    def reset(self):
        """
        Azzera lo stato di tutte le tracce.

        Va richiamato all'inizio di ogni nuova analisi.
        """

        self._states = {}

    def prune(self, active_track_ids):
        """
        Elimina lo stato delle tracce non più presenti nella scena.
        """

        active = set(active_track_ids)

        for track_id in list(self._states):
            if track_id not in active:
                del self._states[track_id]


    def _level_from_distance(self, distance_m, current_level):
        """
        Determina il livello suggerito dalla sola distanza, applicando
        l'isteresi rispetto al livello attualmente confermato.
        """

        # Soglie di rientro più permissive, per isteresi.
        red_exit = self.red_distance_m + self.hysteresis_m
        yellow_exit = self.yellow_distance_m + self.hysteresis_m

        if current_level == self.RED:
            if distance_m <= red_exit:
                return self.RED

            return (
                self.YELLOW
                if distance_m <= yellow_exit
                else self.GREEN
            )

        if current_level == self.YELLOW:
            if distance_m <= self.red_distance_m:
                return self.RED

            return (
                self.YELLOW
                if distance_m <= yellow_exit
                else self.GREEN
            )

        # Livello corrente VERDE: si usano le soglie nominali.
        if distance_m <= self.red_distance_m:
            return self.RED

        if distance_m <= self.yellow_distance_m:
            return self.YELLOW

        return self.GREEN

    def _level_from_ttc(self, ttc_s):
        """
        Determina il livello suggerito dal tempo residuo alla collisione.

        Un valore None, negativo o infinito indica che il pedone non si sta
        avvicinando, e quindi non comporta alcuna allerta.
        """

        if ttc_s is None or ttc_s <= 0:
            return self.GREEN

        if ttc_s <= self.red_ttc_s:
            return self.RED

        if ttc_s <= self.yellow_ttc_s:
            return self.YELLOW

        return self.GREEN

    def _in_path(self, lateral_m, lateral_speed_mps, ttc_s, current_level):
        """
        Indica se il pedone si trova, o si troverà, nel corridoio di marcia.

        Senza una stima laterale utilizzabile si risponde in senso
        prudenziale, cioè come se il pedone fosse sulla traiettoria.
        """

        if lateral_m is None:
            return True

        half_width = self.corridor_half_width_m

        # Corridoio più largo per chi è già segnalato: evita il lampeggio.
        if current_level != self.GREEN:
            half_width += self.corridor_hysteresis_m

        if abs(lateral_m) <= half_width:
            return True

        # Orizzonte della previsione: il tempo che manca all'incontro.
        horizon = self.lateral_horizon_s

        if ttc_s is not None and 0 < ttc_s < horizon:
            horizon = ttc_s

        predicted = lateral_m + (lateral_speed_mps or 0.0) * horizon

        if abs(predicted) <= half_width:
            return True

        # Cambio di segno: nell'intervallo il pedone attraversa il corridoio.
        return lateral_m * predicted < 0

    def _apply_corridor_gate(self, level, in_path):
        """
        Declassa di un livello il pedone che resta fuori dal corridoio.

        Il declassamento è di un solo gradino e non porta direttamente al
        verde: un pedone vicino ma laterale merita comunque attenzione.
        """

        if in_path:
            return level

        if level == self.RED:
            return self.YELLOW

        if level == self.YELLOW:
            return self.GREEN

        return level

    def _apply_confidence_gate(self, level, confidence):
        """
        Declassa il livello quando la confidenza della rilevazione non
        raggiunge la soglia prevista per quel livello.
        """

        if level == self.RED and confidence < self.min_confidence_red:
            level = self.YELLOW

        if level == self.YELLOW and confidence < self.min_confidence_yellow:
            level = self.GREEN

        return level

    def evaluate(
        self,
        track_id,
        distance_m,
        confidence,
        ttc_s=None,
        truncated=False,
        receding=False,
        lateral_m=None,
        lateral_speed_mps=0.0,
    ):
        """
        Restituisce il livello di allerta confermato per una traccia.

        Parametri:
            track_id    identificativo della traccia;
            distance_m  distanza stimata in metri, oppure None;
            confidence  confidenza della rilevazione;
            ttc_s       tempo residuo alla collisione in secondi, o None;
            truncated   True se il bounding box tocca il bordo dell'immagine;
            receding    True se il pedone si sta allontanando;
            lateral_m   scostamento laterale in metri, oppure None;
            lateral_speed_mps  velocità laterale in metri al secondo.
        """

        # Stato attuale della traccia; le tracce nuove partono da VERDE.
        state = self._states.setdefault(
            track_id,
            {
                "level": self.GREEN,
                "pending": None,
                "count": 0,
                "truncated_hold": 0,
            },
        )

        current_level = state["level"]

        # Senza una distanza utilizzabile il livello resta invariato.
        if distance_m is None:
            return current_level

        # Si adotta il più grave fra distanza e TTC.
        distance_level = self._level_from_distance(distance_m, current_level)
        ttc_level = self._level_from_ttc(ttc_s)

        da_ttc = self._SEVERITY[ttc_level] > self._SEVERITY[distance_level]

        proposed = ttc_level if da_ttc else distance_level

        in_path = self._in_path(
            lateral_m,
            lateral_speed_mps,
            ttc_s,
            current_level,
        )

        proposed = self._apply_corridor_gate(proposed, in_path)

        proposed = self._apply_confidence_gate(proposed, confidence)

        # Riquadro troncato: non abbassa il livello, salvo allontanamento o timeout.
        if truncated:
            state["truncated_hold"] += 1
        else:
            state["truncated_hold"] = 0

        blocking = (
            truncated
            and not receding
            and state["truncated_hold"] <= self.max_truncated_hold
            and self._SEVERITY[proposed] < self._SEVERITY[current_level]
        )

        if blocking:
            proposed = current_level

        if proposed == current_level:
            state["pending"] = None
            state["count"] = 0

            return current_level

        if state["pending"] == proposed:
            state["count"] += 1
        else:
            state["pending"] = proposed
            state["count"] = 1

        # Innalzamento rapido, abbassamento lento. Fa eccezione
        # l'innalzamento dovuto al solo tempo al contatto, che richiede
        # conferma su più fotogrammi.
        if self._SEVERITY[proposed] > self._SEVERITY[current_level]:
            required = (
                self.frames_to_escalate_ttc
                if da_ttc
                else self.frames_to_escalate
            )
        else:
            required = self.frames_to_relax

        if state["count"] >= required:
            state["level"] = proposed
            state["pending"] = None
            state["count"] = 0

        return state["level"]

    @classmethod
    def most_severe(cls, levels):
        """
        Restituisce il livello più grave fra quelli forniti.
        """

        worst = cls.GREEN

        for level in levels:
            if cls._SEVERITY.get(level, 0) > cls._SEVERITY[worst]:
                worst = level

        return worst
