"""Source-aware mapped history envelope; owner source data stays opaque here.

No hydrodynamics/FEM imports. Original v1 mapped records retain their schemas
and digests unchanged inside this additive versioned persistence envelope.
"""
from dataclasses import dataclass
import hashlib,json
from .pressure_history import pressure_history_from_dict
from .sea_state import WaveRealization

TYPE='source_aware_mapped_pressure_v1'
WINDOW_TYPE='source_aware_mapped_pressure_v2'
FORCE_TYPE='source_aware_mapped_pressure_v3'


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


@dataclass(frozen=True)
class SourceAwarePressureHistory:
    mapped_history: object
    source_payload_json: str
    realization_json: str
    domain_json: str
    content_sha256: str = ''
    record_times: tuple = ()
    selected_window: tuple = ()
    force_payload_json: str = ''

    def __post_init__(self):
        from .pressure_history import MappedPressureHistory
        from .spectral_pressure_history import SpectralMappedPressureHistory
        if type(self.mapped_history) not in (MappedPressureHistory,SpectralMappedPressureHistory):
            raise ValueError('source-aware history needs a typed nonnested mapped history')
        # v3 keeps the offline source string opaque while reading its compact
        # preview metadata. Owner pressure fields decode only on Apply/remap.
        names=('realization_json','domain_json') if self.force_payload_json else ('source_payload_json','realization_json','domain_json')
        for name in names:
            object.__setattr__(self,name,_json(json.loads(getattr(self,name))))
        if self.force_payload_json:
            if not self.record_times or not self.selected_window:
                raise ValueError('v3 compact force metadata requires explicit full record and selected interval')
            force=json.loads(self.force_payload_json)
            if type(force) is not dict or force.get('version') not in ('excitation_bank_v1','excitation_bank_v2') or not isinstance(self.source_payload_json,str) or not self.source_payload_json:
                raise ValueError('v3 needs compact excitation metadata and an opaque offline owner record')
            object.__setattr__(self,'force_payload_json',_json(force))
        else:
            owner=json.loads(self.source_payload_json)
            if type(owner) is not dict or owner.get('record_type')!='anyhydro_pressure_response_bank_v1':
                raise ValueError('source-aware history needs a canonical owner pressure-bank record')
        WaveRealization.from_dict(json.loads(self.realization_json))
        if self.record_times:
            from .wave_preview import select_sample_window
            window=select_sample_window(self.record_times,*(self.selected_window or (None,None)))
            if self.selected_window and tuple(self.selected_window)!=window.interval:
                raise ValueError('persisted window endpoints must be exact original sample times')
            if tuple(self.mapped_history.times)!=window.times:
                raise ValueError('mapped history times differ from selected original record samples')
            object.__setattr__(self,'record_times',window.record_times)
            object.__setattr__(self,'selected_window',window.interval)
        elif self.selected_window:
            raise ValueError('selected window requires retained full record times')
        expected=hashlib.sha256(_json(self._payload()).encode()).hexdigest()
        if self.content_sha256 and self.content_sha256!=expected:
            raise ValueError('source-aware pressure history content identity mismatch')
        object.__setattr__(self,'content_sha256',expected)

    @property
    def times(self):return self.mapped_history.times
    @property
    def element_ids(self):return self.mapped_history.element_ids
    @property
    def mesh_content_sha256(self):return self.mapped_history.mesh_content_sha256
    @property
    def source_id(self):return self.mapped_history.source_id
    @property
    def provenance(self):return self.mapped_history.provenance
    @property
    def full_record_times(self):return self.record_times or self.times
    @property
    def selected_interval(self):return self.selected_window or (self.times[0],self.times[-1])
    @property
    def real(self):return self.mapped_history.real
    @property
    def imag(self):return self.mapped_history.imag
    @property
    def omega(self):return self.mapped_history.omega

    def _payload(self):
        data=dict(mapped_history=self.mapped_history.to_dict(),source_payload_json=self.source_payload_json,
            realization_json=self.realization_json,domain_json=self.domain_json)
        if self.record_times:data.update(record_times=self.record_times,selected_window=self.selected_window)
        if self.force_payload_json:data['force_payload_json']=self.force_payload_json
        return data

    def to_dict(self):return dict(type=FORCE_TYPE if self.force_payload_json else WINDOW_TYPE if self.record_times else TYPE,**self._payload(),content_sha256=self.content_sha256)

    def compact_preview(self,settings):
        if not self.force_payload_json:raise ValueError('legacy record needs explicit force-cache migration first')
        try:saved=json.loads(self.provenance).get('settings',{})
        except (ValueError,TypeError,AttributeError):saved={}
        if not isinstance(saved,dict) or not saved:
            raise ValueError('compact preview requires saved inputs or certified application restoration')
        if _json({k:v for k,v in settings.items() if k!='name'})!=_json({k:v for k,v in saved.items() if k!='name'}):
            raise ValueError('compact preview inputs differ from saved actual realization')
        return dict(version='retained_force_preview_v1',excitation_bank=json.loads(self.force_payload_json),
            realization=json.loads(self.realization_json),record_times=self.full_record_times,
            settings=settings,source_history_id=self.content_sha256)

    @classmethod
    def from_dict(cls,data):
        data=dict(data)
        kind=data.pop('type',None)
        if kind not in (TYPE,WINDOW_TYPE,FORCE_TYPE) or not data.get('content_sha256'):
            raise ValueError('unsupported or unverified source-aware pressure history')
        if (kind==TYPE and any(key in data for key in ('record_times','selected_window'))) or (
                kind in (WINDOW_TYPE,FORCE_TYPE) and (not data.get('record_times') or not data.get('selected_window'))) or (
                kind==FORCE_TYPE and not data.get('force_payload_json')) or (kind!=FORCE_TYPE and 'force_payload_json' in data):
            raise ValueError('source-aware record/window version mismatch')
        data['mapped_history']=pressure_history_from_dict(data['mapped_history'])
        return cls(**data)

    def evaluate(self,time,element_ids=None):return self.mapped_history.evaluate(time,element_ids)
