import React from 'react';
import {render,screen,fireEvent,cleanup} from '@testing-library/react';
import {it,expect,vi,afterEach} from 'vitest';
import Alert from '../AlertTriggerModal';
afterEach(()=>cleanup());
const props={isOpen:true,symbol:'AAPL',currentPrice:100,optimalEntryMin:99,optimalEntryMax:101,stopLoss:95,takeProfit1:115};
it('reconstructed: named modal dialog',()=>{render(<Alert {...props} onClose={()=>{}}/>);expect(screen.getByRole('dialog',{name:'Execution Price & Invalidation Alerts'}).getAttribute('aria-modal')).toBe('true');});
it('reconstructed: Escape closes',()=>{const close=vi.fn();render(<Alert {...props} onClose={close}/>);fireEvent.keyDown(document,{key:'Escape'});expect(close).toHaveBeenCalledOnce();});
it('reconstructed: focus enters modal',()=>{const trigger=document.createElement('button');document.body.appendChild(trigger);trigger.focus();render(<Alert {...props} onClose={()=>{}}/>);try{expect(document.activeElement).not.toBe(trigger);}finally{trigger.remove();}});
it('reconstructed: invalid levels remain suppressed',()=>{render(<Alert {...props} stopLoss={0} onClose={()=>{}}/>);expect((screen.getByRole('button',{name:'Levels Suppressed'}) as HTMLButtonElement).disabled).toBe(true);});

it('traps focus in both directions and restores the invoking button on close',()=>{const trigger=document.createElement('button');document.body.appendChild(trigger);trigger.focus();const close=vi.fn();const view=render(<Alert {...props} onClose={close}/>);const first=screen.getByRole('button',{name:'Close Trigger Alert'}),last=screen.getByRole('button',{name:'Save Active Alerts'});last.focus();fireEvent.keyDown(last,{key:'Tab'});expect(document.activeElement).toBe(first);fireEvent.keyDown(first,{key:'Tab',shiftKey:true});expect(document.activeElement).toBe(last);view.rerender(<Alert {...props} isOpen={false} onClose={close}/>);expect(document.activeElement).toBe(trigger);trigger.remove();});
