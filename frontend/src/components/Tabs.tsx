interface TabsProps {
  active: "model" | "user";
  onChange: (t: "model" | "user") => void;
  modelName: string | null;
}

export default function Tabs({ active, onChange, modelName }: TabsProps) {
  const modelLabel = modelName ? `Model: ${modelName}` : "Model";
  return (
    <div className="tabs" role="tablist">
      <button
        type="button"
        role="tab"
        aria-selected={active === "model"}
        className={`tabs__tab${active === "model" ? " tabs__tab--active" : ""}`}
        onClick={() => onChange("model")}
      >
        {modelLabel}
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={active === "user"}
        className={`tabs__tab${active === "user" ? " tabs__tab--active" : ""}`}
        onClick={() => onChange("user")}
      >
        User
      </button>
    </div>
  );
}
