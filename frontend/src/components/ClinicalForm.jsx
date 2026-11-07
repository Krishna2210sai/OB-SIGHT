const FACTORS = [
  { name: "diabetes", label: "Diabetes", hint: "Pre-existing or gestational" },
  { name: "hypertension", label: "Hypertension", hint: "Chronic or pregnancy-induced" },
  { name: "preeclampsia", label: "Preeclampsia", hint: "Preeclampsia / eclampsia" },
  { name: "liquor_praecox", label: "Liquor praecox", hint: "Prolonged rupture of membranes" },
  { name: "pyrexia", label: "Maternal pyrexia", hint: "Fever during labour" },
  { name: "meconium", label: "Meconium", hint: "Meconium-stained fluid" },
  { name: "induced", label: "Induced labour", hint: "Labour was induced" }
];

export const EMPTY_FORM = {
  age: "",
  gravidity: "",
  parity: "",
  gest_weeks: "",
  diabetes: false,
  hypertension: false,
  preeclampsia: false,
  liquor_praecox: false,
  pyrexia: false,
  meconium: false,
  induced: false,
  presentation: "1"
};

export const SAMPLE_FORM = {
  age: "31",
  gravidity: "2",
  parity: "1",
  gest_weeks: "39.5",
  diabetes: false,
  hypertension: false,
  preeclampsia: false,
  liquor_praecox: false,
  pyrexia: false,
  meconium: false,
  induced: false,
  presentation: "1"
};

export function validateClinical(form) {
  const errors = {};
  const numeric = [
    ["age", "Maternal age", 12, 60],
    ["gravidity", "Gravidity", 0, 20],
    ["parity", "Parity", 0, 20],
    ["gest_weeks", "Gestational age", 20, 45]
  ];
  for (const [key, label, min, max] of numeric) {
    if (form[key] === "" || form[key] === null || form[key] === undefined) {
      errors[key] = `${label} is required`;
      continue;
    }
    const value = Number(form[key]);
    if (!Number.isFinite(value)) {
      errors[key] = `${label} must be a number`;
    } else if (value < min || value > max) {
      errors[key] = `${label} must be between ${min} and ${max}`;
    }
  }
  if (!errors.gravidity && !errors.parity && Number(form.parity) > Number(form.gravidity)) {
    errors.parity = "Parity cannot exceed gravidity";
  }
  return errors;
}

export function toPayload(form) {
  return {
    age: Number(form.age),
    gravidity: Number(form.gravidity),
    parity: Number(form.parity),
    gest_weeks: Number(form.gest_weeks),
    diabetes: Boolean(form.diabetes),
    hypertension: Boolean(form.hypertension),
    preeclampsia: Boolean(form.preeclampsia),
    liquor_praecox: Boolean(form.liquor_praecox),
    pyrexia: Boolean(form.pyrexia),
    meconium: Boolean(form.meconium),
    induced: Boolean(form.induced),
    presentation: Number(form.presentation)
  };
}

function NumberField({ id, label, hint, value, onChange, min, max, step, error, disabled }) {
  return (
    <div className={`field${error ? " field-invalid" : ""}`}>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        name={id}
        type="number"
        inputMode="decimal"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${id}-error` : hint ? `${id}-hint` : undefined}
        disabled={disabled}
        required
      />
      {hint ? <span className="field-hint" id={`${id}-hint`}>{hint}</span> : null}
      {error ? <span className="field-error" id={`${id}-error`}>{error}</span> : null}
    </div>
  );
}

export default function ClinicalForm({ form, onChange, errors = {}, disabled = false }) {
  const setField = (key) => (value) => onChange({ ...form, [key]: value });

  return (
    <div className="clinical-form">
      <fieldset className="form-section" disabled={disabled}>
        <legend>Maternal and pregnancy data</legend>
        <div className="field-grid">
          <NumberField id="age" label="Maternal age" hint="Years" value={form.age} onChange={setField("age")} min={12} max={60} step={0.1} error={errors.age} disabled={disabled} />
          <NumberField id="gest_weeks" label="Gestational age" hint="Completed weeks" value={form.gest_weeks} onChange={setField("gest_weeks")} min={20} max={45} step={0.1} error={errors.gest_weeks} disabled={disabled} />
          <NumberField id="gravidity" label="Gravidity" hint="Total pregnancies" value={form.gravidity} onChange={setField("gravidity")} min={0} max={20} step={1} error={errors.gravidity} disabled={disabled} />
          <NumberField id="parity" label="Parity" hint="Previous deliveries" value={form.parity} onChange={setField("parity")} min={0} max={20} step={1} error={errors.parity} disabled={disabled} />
          <div className={`field${errors.presentation ? " field-invalid" : ""}`}>
            <label htmlFor="presentation">Fetal presentation</label>
            <select
              id="presentation"
              value={form.presentation}
              onChange={(event) => setField("presentation")(event.target.value)}
              disabled={disabled}
            >
              <option value="1">Vertex</option>
              <option value="2">Breech</option>
              <option value="3">Other / abnormal</option>
            </select>
            {errors.presentation ? <span className="field-error">{errors.presentation}</span> : null}
          </div>
        </div>
      </fieldset>

      <fieldset className="form-section" disabled={disabled}>
        <legend>Intrapartum risk factors</legend>
        <p className="section-note">Tick only what is documented. All variables are used exactly as recorded.</p>
        <div className="checkbox-grid">
          {FACTORS.map((factor) => (
            <label className="check-card" key={factor.name}>
              <input
                type="checkbox"
                checked={Boolean(form[factor.name])}
                onChange={(event) => setField(factor.name)(event.target.checked)}
                disabled={disabled}
              />
              <span>
                <strong>{factor.label}</strong>
                <small>{factor.hint}</small>
              </span>
            </label>
          ))}
        </div>
      </fieldset>
    </div>
  );
}
