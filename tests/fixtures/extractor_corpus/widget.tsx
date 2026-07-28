import type { ReactNode } from "react";

export interface WidgetProps {
  label: string;
  children?: ReactNode;
}

export function Widget({ label, children }: WidgetProps) {
  return (
    <div className="widget">
      <span>{label}</span>
      {children}
    </div>
  );
}

export default Widget;
