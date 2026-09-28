import { Select as SelectPrimitive } from '@base-ui/react/select';
import { Check, ChevronDown } from 'lucide-react';
import { cn } from 'cn';

interface SelectOption {
  value: string;
  label: string;
}

interface SelectFieldProps {
  id?: string;
  'aria-label'?: string;
  className?: string;
  disabled?: boolean;
  value: string;
  options: SelectOption[];
  onValueChange: (value: string) => void;
}

export function SelectField({
  id,
  'aria-label': ariaLabel,
  className,
  disabled,
  value,
  options,
  onValueChange,
}: SelectFieldProps) {
  return (
    <SelectPrimitive.Root
      value={value}
      items={options}
      disabled={disabled}
      onValueChange={(nextValue) => {
        if (nextValue !== null) onValueChange(nextValue);
      }}
    >
      <SelectPrimitive.Trigger
        id={id}
        aria-label={ariaLabel}
        data-slot="select-trigger"
        className={cn(
          'flex h-10 min-w-0 items-center justify-between gap-3 rounded-md border border-input bg-card px-3 text-left text-sm text-foreground outline-none transition-colors hover:bg-muted/50 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50',
          className
        )}
      >
        <SelectPrimitive.Value className="min-w-0 truncate" />
        <SelectPrimitive.Icon className="shrink-0 text-muted-foreground">
          <ChevronDown className="size-4" />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Positioner
          side="bottom"
          sideOffset={4}
          align="start"
          alignItemWithTrigger={false}
          className="isolate z-50"
        >
          <SelectPrimitive.Popup
            data-slot="select-popup"
            className="max-h-[min(18rem,var(--available-height))] min-w-[var(--anchor-width)] max-w-[calc(100vw-2rem)] overflow-y-auto rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-md outline-none"
          >
            <SelectPrimitive.List>
              {options.map((option) => (
                <SelectPrimitive.Item
                  key={option.value}
                  value={option.value}
                  className="flex min-h-9 cursor-default items-center justify-between gap-3 rounded-sm px-2.5 py-1.5 text-sm outline-none data-highlighted:bg-accent data-highlighted:text-accent-foreground data-selected:font-medium"
                >
                  <SelectPrimitive.ItemText className="min-w-0 truncate">
                    {option.label}
                  </SelectPrimitive.ItemText>
                  <SelectPrimitive.ItemIndicator className="shrink-0 text-foreground">
                    <Check className="size-4" />
                  </SelectPrimitive.ItemIndicator>
                </SelectPrimitive.Item>
              ))}
            </SelectPrimitive.List>
          </SelectPrimitive.Popup>
        </SelectPrimitive.Positioner>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}
