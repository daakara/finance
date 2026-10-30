import {describe,it,expect} from "vitest";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

// Execute the actual page callback with controlled arrivals, not a copied guard implementation.
const source=fs.readFileSync("app/page.tsx","utf8");
const tree=ts.createSourceFile("page.tsx",source,ts.ScriptTarget.Latest,true,ts.ScriptKind.TSX);
let retry="";
function visit(node:ts.Node){if(ts.isJsxAttribute(node)&&node.name.getText(tree)==="onClick"&&node.initializer&&ts.isJsxExpression(node.initializer)){const e=node.initializer.expression;if(e&&ts.isArrowFunction(e)&&e.getText(tree).includes("cacheRef.current.clear()")&&e.getText(tree).includes(".then((res)"))retry=e.getText(tree);}ts.forEachChild(node,visit);}
visit(tree);
function run(){let resolve!:(x:unknown)=>void,reject!:(x:unknown)=>void;const promise=new Promise((a,b)=>{resolve=a;reject=b;});const state={data:null as unknown,error:null as unknown,loading:false};const token={current:0};const scope={exports:{} as {handler:()=>void},analysisRequestIdRef:token,cacheRef:{current:new Map()},selectedSymbol:"AAPL",userRole:"LONG_TERM",setData:(x:unknown)=>state.data=x,setError:(x:unknown)=>state.error=x,setLoading:(x:boolean)=>state.loading=x,fetchAssetAnalytics:()=>promise,Error};const js=ts.transpileModule("exports.handler="+retry,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText;vm.runInNewContext(js,scope);scope.exports.handler();return {state,token,resolve,reject};}
const flush=()=>new Promise(r=>setTimeout(r,0));
describe("Analysis response generation",()=>{
 it("keeps a newer blocked decision when obsolete Retry succeeds",async()=>{expect(retry).not.toBe("");const x=run();const current={symbol:"NVDA",isActionable:false};x.token.current++;x.state.data=current;x.resolve({symbol:"AAPL",isActionable:true});await flush();expect(x.state.data).toBe(current);});
 it("does not clear the newer loading/error state when obsolete Retry rejects",async()=>{const x=run();x.token.current++;x.state.error="new context";x.state.loading=true;x.reject(new Error("old failure"));await flush();expect(x.state.error).toBe("new context");expect(x.state.loading).toBe(true);});
 it("accepts the current Retry without changing its financial flags",async()=>{const x=run();const current={symbol:"AAPL",isActionable:false};x.resolve(current);await flush();expect(x.state.data).toBe(current);expect(x.state.loading).toBe(false);});
});
