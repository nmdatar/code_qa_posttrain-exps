import type { Model } from './types';
export default function HarnessSelect({value,onChange,model,disabled,label='Tools'}: {value:string;onChange:(value:string)=>void;model?:Model;disabled?:boolean;label?:string}) {
  const recommended = model?.trained_harness === 'bash' ? 'Bash' : 'Structured tools';
  return <label className="compare-picker"><span>{label}</span><select aria-label={label} value={value} disabled={disabled} onChange={e=>onChange(e.target.value)}>
    <option value="auto">Auto · {recommended}</option>
    <option value="structured">Structured tools</option><option value="bash">Bash only</option>
  </select><small>{model?.trained_harness ? `Post-trained with ${recommended.toLowerCase()}` : 'Choose the tools available to this model'}</small></label>;
}
