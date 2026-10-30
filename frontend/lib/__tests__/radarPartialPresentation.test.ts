import {describe,it,expect,vi,afterEach} from "vitest";
import {fetchScreenerGems} from "../api";
afterEach(()=>vi.unstubAllGlobals());
describe("Radar hosted contract presentation",()=>{
 it("preserves partial results and their incomplete status",async()=>{vi.stubGlobal("fetch",vi.fn(async()=>({ok:true,json:async()=>({status:"PARTIAL",scanState:"PARTIAL",datasetState:"CANDIDATES_AVAILABLE",partialErrors:[{ticker:"AAPL"}],results:[],error:"One evaluation failed"})})));const r=await fetchScreenerGems();expect(r.status).toBe("PARTIAL");expect(r.scanState).toBe("PARTIAL");expect(r.partialErrors).toHaveLength(1);expect(r.error).toBe("One evaluation failed");});
 it("does not promote an HTTP-200 scan failure to success",async()=>{vi.stubGlobal("fetch",vi.fn(async()=>({ok:true,json:async()=>({status:"ERROR",datasetState:"SCAN_FAILED",results:[],error:"Unavailable"})})));const r=await fetchScreenerGems();expect(r.status).toBe("ERROR");expect(r.datasetState).toBe("SCAN_FAILED");});
});
